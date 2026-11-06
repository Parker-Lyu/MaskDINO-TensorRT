"""Prepare native input manifests and use the existing COCO evaluation and alignment rules."""
import argparse
import ctypes as ct
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOPanopticEvaluator
from stage2.run import postprocess_batch, prepare_batch
from stage4.check_restore import Geometry
from stage3.alignment import LIMITS, panoptic_metrics
from stage3.trt_backend import Engine

SIZE = 1024
SAMPLE = Path(__file__).resolve().parents[1] / "stage2/sample.json"
DATASET = Path("/coco")


def load_requests(mode):
    annotation = DATASET / "annotations/panoptic_val2017.json"
    data = json.loads(annotation.read_text())
    images = {image["id"]: image for image in data["images"]}
    expected = {ann["image_id"] for ann in data["annotations"]}
    assert len(images) == len(expected) == len(data["annotations"]) == len(data["images"]) == 5000
    assert set(images) == expected
    sample = json.loads(SAMPLE.read_text())
    ids = (sorted(images) if mode == "evaluate" else
           sample["image_ids"] if mode == "benchmark" else sample["smoke_image_ids"])
    assert len(ids) == len(set(ids)) == {"align": 32, "evaluate": 5000, "benchmark": 500, "trace": 32}[mode]
    return annotation, [images[i] for i in ids]


def image_path(image):
    return DATASET / "val2017" / image["file_name"]


def native_prediction(output, image):
    stem = output / "predictions" / str(image["id"])
    ids = np.fromfile(stem.with_suffix(".bin"), dtype=np.int32).reshape(image["height"], image["width"])
    segments = []
    for line in stem.with_suffix(".tsv").read_text().splitlines():
        segment, category, thing = map(int, line.split())
        segments.append(dict(id=segment, category_id=category, isthing=bool(thing)))
    assert set(np.unique(ids)) - {0} == {s["id"] for s in segments}
    return {"panoptic_seg": (torch.from_numpy(ids), segments)}


class Segment(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ("id", "category_id", "isthing")]


class Segments(ct.Structure):
    _fields_ = [("count", ct.c_int), ("items", Segment * 300)]


class NativePanoptic:
    def __init__(self, library, model):
        self.run = ct.CDLL(str(library)).stage4_panoptic
        self.run.argtypes = [ct.c_void_p] * 3 + [Geometry] + [ct.c_void_p] * 3
        self.things = torch.zeros(133, dtype=torch.int32, device="cuda")
        self.things[list(model.metadata.thing_dataset_id_to_contiguous_id.values())] = 1
        torch.cuda.synchronize()

    def __call__(self, logits, masks, geometry):
        result_map = np.empty((geometry.height, geometry.width), dtype=np.int32)
        segments = Segments()
        status = self.run(logits.data_ptr(), masks.data_ptr(), self.things.data_ptr(), geometry,
                          torch.cuda.current_stream().cuda_stream, result_map.ctypes.data, ct.byref(segments))
        assert status == 0
        metadata = [dict(id=s.id, category_id=s.category_id, isthing=bool(s.isthing))
                    for s in segments.items[:segments.count]]
        return {"panoptic_seg": (torch.from_numpy(result_map), metadata)}


def check_boundaries(model, native):
    geometry = Geometry(1, 5, 205, 1024)
    checked = {}

    def verify(name, logits, masks, expected_ids, expected_categories):
        reference = postprocess_batch(
            model, [dict(id=name, height=1, width=5)], [(None, (205, 1024), (1, 5))],
            {"pred_logits": logits, "pred_masks": masks}, SIZE, time.perf_counter(), False,
        )[0][0]["panoptic_seg"]
        result = native(logits, masks, geometry)["panoptic_seg"]
        assert result[0].tolist() == reference[0].tolist() == [expected_ids], name
        assert result[1] == reference[1], name
        assert [segment["category_id"] for segment in result[1]] == expected_categories, name
        checked[name] = dict(map=expected_ids, metadata=result[1])

    logits = torch.full((1, 300, 133), -100.0, device="cuda")
    masks = torch.zeros((1, 300, 256, 256), device="cuda")
    verify("no_candidates", logits, masks, [0] * 5, [])
    logits[0, 0, 132] = -1.0986123085021973  # FP32 logit(0.25)
    assert logits[0, 0, 132].sigmoid().item() == 0.25
    verify("score_equal_to_threshold", logits, masks, [0] * 5, [])
    logits[0, 0, 132] = -1.09861
    verify("mask_equal_to_threshold_and_class_132", logits, masks, [1] * 5, [132])

    logits[0, 0, 132] = logits[0, 1, 0] = 0.0
    verify("winner_tie_uses_first_query", logits, masks, [1] * 5, [132])
    masks[0, 1].fill_(-20.0)
    masks[0, 1, :, :51] = 20.0
    verify("overlap_equal_to_threshold", logits, masks, [2, 1, 1, 1, 1], [132, 0])
    masks[0, 1, :, :102] = 20.0
    verify("overlap_below_threshold", logits, masks, [1, 1, 0, 0, 0], [0])

    logits[0, 1, 0] = -100.0
    logits[0, 1, 132] = 0.0
    masks[0, 0].copy_(-masks[0, 1])
    verify("merge_same_stuff_category", logits, masks, [1] * 5, [132])
    return checked


@torch.inference_mode()
def align(output, engine, images):
    from stage2.run import load_model

    model, _ = load_model("/checkpoint.pth")
    model.backbone.cpu()
    model.sem_seg_head.cpu()
    torch.cuda.empty_cache()
    panoptic = NativePanoptic(output / "restore-check.so", model)
    boundary_checks = check_boundaries(model, panoptic)
    observations = []
    limits = LIMITS["fp32"]
    with torch.cuda.stream(torch.cuda.Stream()):
        network = Engine(engine)
        for image in images:
            inputs, geometry = prepare_batch(model, [image_path(image).read_bytes()], SIZE)
            raw = network(inputs)
            reference = postprocess_batch(model, [image], geometry, raw, SIZE, time.perf_counter(), False)[0][0]
            _, valid, original = geometry[0]
            isolated = panoptic(raw["pred_logits"], raw["pred_masks"], Geometry(*original, *valid))
            isolated_metrics, _ = panoptic_metrics(reference, isolated)
            actual = native_prediction(output, image)
            metrics, diff = panoptic_metrics(reference, actual)
            passed = all(
                result["category_agreement"] >= limits["category_agreement"]
                and not any(result["unmatched_segments"])
                and min((match["iou"] for match in result["same_category_matches"]), default=1) >= limits["segment_iou"]
                for result in (metrics, isolated_metrics)
            )
            observations.append(dict(image_id=image["id"], passed=passed, panoptic=metrics,
                                     same_tensor_panoptic=isolated_metrics))
            print(image["id"], metrics["category_agreement"], isolated_metrics["category_agreement"], flush=True)
            if not passed:
                Image.fromarray(diff.astype(np.uint8) * 255).save(output / f"anomaly-{image['id']}.png")
                torch.save({"reference": reference, "native": actual, "isolated": isolated,
                            "raw": {key: tensor.cpu() for key, tensor in raw.items()},
                            "geometry": [*original, *valid]}, output / f"anomaly-{image['id']}.pt")
    return {"per_image": observations, "boundary_checks": boundary_checks,
            "passed": all(row["passed"] for row in observations)}


def evaluate(output, annotation, images):
    meta = MetadataCatalog.get("coco_2017_val_panoptic_with_sem_seg")
    name = "stage4_coco_panoptic_val"
    MetadataCatalog.get(name).set(
        thing_dataset_id_to_contiguous_id=meta.thing_dataset_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id=meta.stuff_dataset_id_to_contiguous_id,
        panoptic_json=str(annotation),
        panoptic_root=str(DATASET / "panoptic_val2017"),
    )
    evaluator = COCOPanopticEvaluator(name, str(output))
    evaluator.reset()
    for image in images:
        evaluator.process([image], [native_prediction(output, image)])
    scores = evaluator.evaluate()["panoptic_seg"]
    covered = [p["image_id"] for p in json.loads((output / "predictions.json").read_text())["annotations"]]
    assert len(covered) == 5000 and set(covered) == {image["id"] for image in images}
    baseline = json.loads((SAMPLE.parents[1] / "stage3/results/fp32-pq-rerun/result.json").read_text())["scores"]
    deltas = {key: scores[key] - baseline[key] for key in ("PQ", "PQ_th", "PQ_st")}
    return dict(scores=scores, unique_prediction_ids=len(set(covered)), delta_vs_stage3=deltas,
                passed=all(delta >= -LIMITS["fp32"]["pq_delta"] for delta in deltas.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("manifest", "report"))
    parser.add_argument("mode", choices=("align", "evaluate", "benchmark", "trace"))
    parser.add_argument("output", type=Path)
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    annotation, images = load_requests(args.mode)
    if args.action == "manifest":
        lines = [f'{im["id"]}\t{im["height"]}\t{im["width"]}\t{json.dumps(str(image_path(im)))}\n' for im in images]
        (args.output / "manifest.tsv").write_text("".join(lines))
        things = MetadataCatalog.get("coco_2017_val_panoptic_with_sem_seg").thing_dataset_id_to_contiguous_id.values()
        (args.output / "things.txt").write_text(" ".join(map(str, sorted(things))) + "\n")
        return

    first, *rows = (args.output / "native.tsv").read_text().splitlines()
    count, seconds, peak = first.split()
    assert int(count) == len(images)
    with args.engine.open("rb") as file:
        engine_hash = file_digest(file, "sha256").hexdigest()
    report = dict(mode=args.mode, backend="native", images=int(count), failures=0, size=SIZE,
                  batch=1, slots=1, precision="fp32", tf32=False, engine_sha256=engine_hash,
                  loop_seconds=float(seconds), cuda_stream="dedicated non-default stream")
    if args.mode == "align":
        report.update(align(args.output, args.engine, images))
    if args.mode == "evaluate":
        report.update(evaluate(args.output, annotation, images))
    if args.mode == "benchmark":
        samples = [dict(image_id=int(i), network_ms=float(network), e2e_ms=float(e2e))
                   for i, network, e2e in (row.split() for row in rows)]
        assert [row["image_id"] for row in samples] == [image["id"] for image in images]
        report.update(warmup=20, samples=samples, images_per_second=len(images) / float(seconds),
                      memory_scope="NVML CUDA compute process (host PID)", memory_sample_interval_ms=20,
                      sampled_peak_process_used_bytes=int(peak))
        for field in ("network_ms", "e2e_ms"):
            for percentile, value in zip((50, 95), np.percentile([s[field] for s in samples], [50, 95])):
                report[f"{field}_p{percentile}"] = float(value)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print({key: value for key, value in report.items() if key not in ("samples", "per_image")}, flush=True)
    if not report.get("passed", True):
        raise RuntimeError(f"Stage 4 {args.mode} gate failed")


if __name__ == "__main__":
    main()
