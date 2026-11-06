"""Verify Stage 3 coverage and summarize the three benchmark rounds."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

SAMPLE = Path(__file__).resolve().parents[1] / "stage2/sample.json"
METRICS = ("network_ms_p50", "network_ms_p95", "e2e_ms_p50", "e2e_ms_p95", "images_per_second",
           "sampled_peak_device_used_bytes")
SCORES = ("PQ", "PQ_th", "PQ_st")
PQ_LIMITS = {"fp32": 0.05, "mixed": 0.20}


def pq_deltas(reference, candidate):
    assert reference["unique_prediction_ids"] == candidate["unique_prediction_ids"] == 5000
    delta = {key: candidate["scores"][key] - reference["scores"][key] for key in SCORES}
    return delta, all(value >= -PQ_LIMITS[candidate["backend"]] for value in delta.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("benchmarks", type=Path, nargs="*")
    args = parser.parse_args()
    reference, candidate = [json.loads(path.read_text()) for path in (args.reference, args.candidate)]
    delta, passed = pq_deltas(reference, candidate)
    report = {"backend": candidate["backend"], "pq_delta": delta, "pq_passed": passed}
    groups = {}
    ids = json.loads(SAMPLE.read_text())["image_ids"]
    for path in args.benchmarks:
        data = json.loads(path.read_text())
        assert data["mode"] == "benchmark" and data["warmup"] == 20 and data["failures"] == 0, path
        assert (data["size"], data["batch"], data["slots"], data["images"]) == (1024, 1, 1, 500), path
        assert [item["image_id"] for item in data["samples"]] == ids, path
        assert all(math.isfinite(data[key]) and data[key] > 0 for key in METRICS), path
        groups.setdefault(data["backend"], []).append(data)
    report["performance"] = {}
    for backend, rounds in groups.items():
        assert len(rounds) == 3 and len({data["engine_sha256"] for data in rounds}) == 1
        statistics = {}
        for key in METRICS:
            values = [data[key] for data in rounds]
            statistics[key] = {"median": median(values), "min": min(values), "max": max(values)}
        report["performance"][backend] = statistics
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
