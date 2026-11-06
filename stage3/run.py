"""Fixed 1024/B1/single-slot comparison using the Stage 2 pipeline."""
import argparse
from contextlib import nullcontext
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch

from stage2.run import (
    geometry_check, load_model, make_evaluator, postprocess_batch,
    prepare_batch, pytorch_network, run_batch, setup_logger,
)

SIZE = 1024
DATASET = Path("/coco")
OUTPUT = Path("/run-output")
SAMPLE = Path(__file__).resolve().parents[1] / "stage2/sample.json"


def dump(name, data):
    (OUTPUT / name).write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "build", "plugin-test", "align", "evaluate", "benchmark", "trace"))
    parser.add_argument("--backend", choices=("pytorch", "fp32", "mixed"), default="pytorch")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--cpu-constants", action="store_true")
    args = parser.parse_args()
    if args.mode in ("build", "plugin-test"):
        from stage3 import trt_backend
        if args.mode == "build":
            trt_backend.build(args.onnx, args.backend)
        else:
            from stage3.plugin_test import validate
            validate()
        return
    setup_logger()
    geometry_check(SIZE)
    annotation = DATASET / "annotations/panoptic_val2017.json"
    data = json.loads(annotation.read_text())
    records = {record["id"]: record for record in data["images"]}
    expected = {ann["image_id"] for ann in data["annotations"]}
    assert len(records) == len(data["images"]) == len(expected) == len(data["annotations"]) == 5000
    assert set(records) == expected
    sample = json.loads(SAMPLE.read_text())
    assert len(sample["smoke_image_ids"]) == 32 and len(sample["image_ids"]) == 500
    ids = sorted(records) if args.mode == "evaluate" else (
        sample["image_ids"] if args.mode == "benchmark" else sample["smoke_image_ids"]
    )
    assert len(ids) == len(set(ids)) and set(ids) <= expected

    def read(image_id):
        return (DATASET / "val2017" / records[image_id]["file_name"]).read_bytes()

    model, cfg = load_model("/checkpoint.pth")
    (OUTPUT / "resolved-config.yaml").write_text(cfg.dump())
    if args.mode == "export":
        from stage3.trt_backend import export
        image, _ = prepare_batch(model, [read(ids[0])], SIZE)
        export(model, image, args.backend, args.cpu_constants)
        return
    network = None
    engine_hash = None
    if args.backend != "pytorch":
        from stage3.trt_backend import Engine
        network = Engine(args.engine)
        engine_hash = digest(args.engine)
        # Postprocessing uses model methods and normalization buffers only.
        if args.mode != "align":
            del model.backbone, model.sem_seg_head
            torch.cuda.empty_cache()
    report = {
        "mode": args.mode, "backend": args.backend, "size": SIZE, "batch": 1, "slots": 1,
        "torch": torch.__version__, "cuda": torch.version.cuda, "tf32": False,
        "sample_sha256": digest(SAMPLE), "config_sha256": digest(OUTPUT / "resolved-config.yaml"),
        "engine_sha256": engine_hash, "images": len(ids), "failures": 0,
    }
    if args.mode == "align":
        from stage3.alignment import validate
        validate(model, network, records, ids, read, args.backend, report)
        return

    evaluator = None
    if args.mode == "evaluate":
        evaluator = make_evaluator(model, cfg, annotation, DATASET, OUTPUT)
    benchmark = args.mode == "benchmark"
    trace = args.mode == "trace"
    cached = {i: read(i) for i in ids} if benchmark or trace else {}
    region = torch.cuda.nvtx.range if trace else nullcontext

    if cached:
        for _ in range(20):
            run_batch(model, [records[ids[0]]], [cached[ids[0]]], SIZE, network=network)
        report["warmup"] = 20

    from stage3.memory import GpuMemory
    memory = GpuMemory() if benchmark else nullcontext()
    samples = []
    if trace:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        with memory:
            start = time.perf_counter()
            for offset, image_id in enumerate(ids, 1):
                jpeg = cached[image_id] if cached else read(image_id)
                with region(f"image {image_id}"):
                    predictions, _, elapsed, network_ms = run_batch(
                        model, [records[image_id]], [jpeg], SIZE,
                        timing=benchmark, network=network, region=region,
                    )
                if evaluator:
                    evaluator.process([{"image_id": image_id, "file_name": records[image_id]["file_name"]}], predictions)
                    if offset % 100 == 0:
                        print(f"Evaluated {offset}/{len(ids)} images")
                if benchmark:
                    samples.append({"image_id": image_id, "network_ms": network_ms, "e2e_ms": elapsed[0]})
            report["loop_seconds"] = time.perf_counter() - start
    except Exception:
        report["failures"] += 1
        report["completed_images"] = offset - 1
        dump("result.json", report)
        raise
    finally:
        if trace:
            torch.cuda.cudart().cudaProfilerStop()
    if evaluator:
        report["scores"] = evaluator.evaluate()["panoptic_seg"]
        covered = [p["image_id"] for p in json.loads((OUTPUT / "predictions.json").read_text())["annotations"]]
        assert len(covered) == 5000 and set(covered) == expected
        report["unique_prediction_ids"] = len(set(covered))
    if benchmark:
        report.update(memory.results())
        report["images_per_second"] = len(ids) / report["loop_seconds"]
        for field in ("network_ms", "e2e_ms"):
            for percentile, value in zip((50, 95), np.percentile([s[field] for s in samples], [50, 95])):
                report[f"{field}_p{percentile}"] = float(value)
        report["samples"] = samples
    dump("result.json", report)
    print({key: value for key, value in report.items() if key != "samples"})


if __name__ == "__main__":
    main()
