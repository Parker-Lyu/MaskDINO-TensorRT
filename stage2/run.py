"""Evaluate and benchmark fixed-square MaskDINO panoptic inference."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.transforms import ResizeTransform
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.logger import setup_logger
from maskdino import add_maskdino_config


def resize_pad(rgb, size):
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    valid = (max(1, round(height * scale)), max(1, round(width * scale)))
    resized = ResizeTransform(height, width, *valid).apply_image(rgb)
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    canvas[:valid[0], :valid[1]] = resized
    return canvas, valid


def prepare(jpeg, size):
    with Image.open(BytesIO(jpeg)) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    canvas, valid = resize_pad(rgb, size)
    return torch.from_numpy(canvas).permute(2, 0, 1), valid, rgb.shape[:2]


def geometry_check(size):
    for shape in ((51, 101), (101, 51), (101, 101)):
        canvas, valid = resize_pad(np.full((*shape, 3), 37, dtype=np.uint8), size)
        assert max(valid) == size and np.all(canvas[:valid[0], :valid[1]] == 37)
        assert np.all(canvas[valid[0]:] == 128) and np.all(canvas[:, valid[1]:] == 128)
        logits = torch.full((1, size, size), -3.0)
        logits[:, :valid[0], :valid[1]] = 2.0
        restored = sem_seg_postprocess(logits, valid, *shape)
        assert restored.shape == (1, *shape)
        torch.testing.assert_close(restored, torch.full_like(restored, 2))


def run_batch(model, records, jpegs, size, timing=False, capture=False):
    start = time.perf_counter()
    inputs = [prepare(jpeg, size) for jpeg in jpegs]
    images = torch.stack([image for image, _, _ in inputs]).to(model.device)
    images = (images - model.pixel_mean) / model.pixel_std
    if timing:
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
    outputs, _ = model.sem_seg_head(model.backbone(images))
    if timing:
        end.record()
    logits, masks = outputs["pred_logits"], outputs["pred_masks"]
    raw = []
    predictions = []
    elapsed = []
    for record, (_, valid, original), cls, mask in zip(records, inputs, logits, masks):
        if original != (record["height"], record["width"]):
            raise ValueError(f"Decoded size {original} disagrees with COCO image {record['id']}")
        if capture:
            raw.append((cls.cpu(), mask.cpu()))
        square = F.interpolate(mask[None], (size, size), mode="bilinear", align_corners=False)[0]
        restored = sem_seg_postprocess(square, valid, *original)
        panoptic_map, segments = model.panoptic_inference(cls, restored)
        predictions.append({"panoptic_seg": (panoptic_map.cpu(), segments)})
        elapsed.append((time.perf_counter() - start) * 1000)
        del square, restored, panoptic_map
    network_ms = begin.elapsed_time(end) if timing else None
    return predictions, raw, elapsed, network_ms


def describe_difference(reference, current):
    ref_raw, ref_output = reference
    raw, output = current
    result = {}
    for name, expected, actual in zip(("class", "mask"), ref_raw, raw):
        delta = (expected - actual).abs()
        result[f"{name}_max_abs"] = delta.max().item()
        result[f"{name}_mean_abs"] = delta.mean().item()
    ref_map, ref_segments = ref_output["panoptic_seg"]
    pan_map, segments = output["panoptic_seg"]
    result["segments_equal"] = ref_segments == segments
    result["changed_pixel_fraction"] = (ref_map != pan_map).float().mean().item()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "evaluate", "benchmark"))
    parser.add_argument("--size", type=int, choices=(640, 800, 1024), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/coco"))
    parser.add_argument("--checkpoint", default="/checkpoint.pth")
    parser.add_argument("--output", type=Path, default=Path("/run-output"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    setup_logger()
    annotation = args.dataset / "annotations/panoptic_val2017.json"
    data = json.loads(annotation.read_text())
    records = {image["id"]: image for image in data["images"]}
    expected = {ann["image_id"] for ann in data["annotations"]}
    assert len(records) == len(data["images"]) == len(expected) == 5000
    assert set(records) == expected
    sample = json.loads(Path(__file__).with_name("sample.json").read_text())
    ids = {
        "probe": sample["smoke_image_ids"],
        "evaluate": sorted(records),
        "benchmark": sample["image_ids"],
    }[args.mode]
    assert len(ids) == len(set(ids)) and set(ids) <= expected
    geometry_check(args.size)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(
        "configs/coco/panoptic-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml"
    )
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.WEIGHTS = args.checkpoint
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    evaluator = None
    if args.mode == "evaluate":
        meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        assert meta.thing_dataset_id_to_contiguous_id == model.metadata.thing_dataset_id_to_contiguous_id
        evaluation_meta = MetadataCatalog.get("stage2_coco_panoptic_val")
        evaluation_meta.set(
            thing_dataset_id_to_contiguous_id=meta.thing_dataset_id_to_contiguous_id,
            stuff_dataset_id_to_contiguous_id=meta.stuff_dataset_id_to_contiguous_id,
            panoptic_json=str(annotation),
            panoptic_root=str(args.dataset / "panoptic_val2017"),
        )
        evaluator = COCOPanopticEvaluator(evaluation_meta.name, str(args.output))
        evaluator.reset()

    def read(image_id):
        return (args.dataset / "val2017" / records[image_id]["file_name"]).read_bytes()

    cached = {image_id: read(image_id) for image_id in ids} if args.mode == "benchmark" else {}
    report = {
        "mode": args.mode, "size": args.size, "batch": args.batch, "images": len(ids),
        "precision": "FP32", "autocast": False, "matmul_tf32": False, "cudnn_tf32": False,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "fallback_enabled": False,
    }
    observations = {}
    batch_samples = []
    with torch.inference_mode():
        if cached:
            warmup_ids = ids[:args.batch]
            for _ in range(args.warmup):
                run_batch(model, [records[i] for i in warmup_ids], [cached[i] for i in warmup_ids], args.size)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for offset in range(0, len(ids), args.batch):
            batch_ids = ids[offset:offset + args.batch]
            batch_records = [records[i] for i in batch_ids]
            jpegs = [cached[i] for i in batch_ids] if cached else [read(i) for i in batch_ids]
            predictions, raw, elapsed, network_ms = run_batch(
                model, batch_records, jpegs, args.size, timing=bool(cached), capture=args.mode == "probe"
            )
            if evaluator:
                inputs = [{"image_id": i, "file_name": records[i]["file_name"]} for i in batch_ids]
                evaluator.process(inputs, predictions)
                if (offset // args.batch + 1) % 100 == 0:
                    print(f"Evaluated {offset + len(batch_ids)}/{len(ids)} images", flush=True)
            if args.mode == "probe":
                observations.update(zip(batch_ids, zip(raw, predictions)))
            if cached:
                batch_samples.append({
                    "image_ids": batch_ids, "e2e_ms": elapsed, "network_ms": network_ms
                })
        report["loop_seconds"] = time.perf_counter() - start
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if args.mode == "probe":
        if args.batch == 1:
            torch.save(observations, args.output / "reference.pt")
        else:
            reference = torch.load(args.reference, weights_only=False)
            report["per_image"] = {
                i: describe_difference(reference[i], observations[i]) for i in ids
            }
    if evaluator:
        report["scores"] = evaluator.evaluate()["panoptic_seg"]
        predictions_json = json.loads((args.output / "predictions.json").read_text())
        covered = [entry["image_id"] for entry in predictions_json["annotations"]]
        assert len(covered) == len(expected) and set(covered) == expected
        report["unique_prediction_ids"] = len(set(covered))
    if cached:
        report["warmup_batches"] = args.warmup
        report["samples"] = batch_samples
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print({key: val for key, val in report.items() if key not in ("per_image", "samples")})


if __name__ == "__main__":
    main()
