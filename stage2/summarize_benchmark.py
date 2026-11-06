"""Summarize fixed-sample benchmark result files, one record per round."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(path, expected_ids, prefix_images):
    result = json.loads(path.read_text())
    samples = result["samples"]
    ids = [image_id for sample in samples for image_id in sample["image_ids"]]
    if result["mode"] != "benchmark" or result["images"] != len(ids) or ids != expected_ids:
        raise ValueError(f"{path}: benchmark image coverage or order differs from sample.json")
    if any(
        len(sample["image_ids"]) != len(sample["e2e_ms"])
        or len(sample["image_ids"]) != min(result["batch"], len(ids) - index * result["batch"])
        for index, sample in enumerate(samples)
    ):
        raise ValueError(f"{path}: inconsistent batch or latency sample size")
    loop_seconds = result["loop_seconds"]
    if prefix_images is not None:
        if not 0 < prefix_images <= len(ids) or prefix_images % result["batch"]:
            raise ValueError(f"{path}: prefix must consist of complete batches")
        samples = samples[:prefix_images // result["batch"]]
        loop_seconds = samples[-1]["loop_elapsed_seconds"]
        ids = ids[:prefix_images]
    network = [sample["network_ms"] for sample in samples]
    e2e = [latency for sample in samples for latency in sample["e2e_ms"]]
    batch_e2e = [sample["e2e_ms"][-1] for sample in samples]
    if not all(math.isfinite(value) and value > 0 for value in [*network, *e2e, loop_seconds]):
        raise ValueError(f"{path}: invalid timing")
    report = {
        "file": str(path),
        "size": result["size"],
        "batch": result["batch"],
        "images": len(ids),
        "warmup_batches": result["warmup_batches"],
        "network_p50_ms": percentile(network, 0.5),
        "network_p95_ms": percentile(network, 0.95),
        "e2e_p50_ms": percentile(e2e, 0.5),
        "e2e_p95_ms": percentile(e2e, 0.95),
        "batch_e2e_p50_ms": percentile(batch_e2e, 0.5),
        "batch_e2e_p95_ms": percentile(batch_e2e, 0.95),
        "images_per_s": len(ids) / loop_seconds,
    }
    if prefix_images is None:
        report.update({
            "peak_allocated_gib": result["peak_allocated_bytes"] / 2**30,
            "peak_reserved_gib": result["peak_reserved_bytes"] / 2**30,
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--prefix-images", type=int, help="Summarize a prefix of complete batches")
    args = parser.parse_args()
    expected_ids = json.loads(Path(__file__).with_name("sample.json").read_text())["image_ids"]
    groups = {}
    for path in args.results:
        report = summarize(path, expected_ids, args.prefix_images)
        groups.setdefault((report["size"], report["batch"]), []).append(report)

    output = []
    for (size, batch), rounds in sorted(groups.items()):
        metrics = [key for key in rounds[0] if key.endswith(("_ms", "_gib", "_per_s"))]
        statistics = {}
        for metric in metrics:
            values = [run[metric] for run in rounds]
            statistics[metric] = {"median": median(values), "min": min(values), "max": max(values)}
        output.append({"size": size, "batch": batch, "rounds": rounds, "statistics": statistics})
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
