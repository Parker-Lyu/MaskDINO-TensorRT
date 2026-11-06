"""Find verified Stage 5 online capacity at each measured P99 budget."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def verified_points(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["arrival_window_seconds"] < 300 or row["completed"] + row["rejected"] < 10000:
            continue
        key = (row["shape"], row["config"], row["delay_ms"], row["rate"])
        seed = row["seed"]
        if seed in groups[key]:
            raise ValueError(f"Duplicate seed at {key}: {seed}")
        groups[key][seed] = row
    points = defaultdict(list)
    for (shape, config, delay, rate), seeds in sorted(groups.items()):
        if len(seeds) < 3:
            continue
        rounds = list(seeds.values())
        points[shape, config, delay].append({
            "rate": rate,
            "seeds": sorted(seeds),
            "p99_ms_by_seed": {seed: row["latency_ms"]["p99"] for seed, row in seeds.items()},
            "max_p99_ms": max(row["latency_ms"]["p99"] for row in rounds),
            "sustained": all(row["sustained"] and row["rejected"] == 0 for row in rounds),
            "peak_bytes": max(row["sampled_peak_process_used_bytes"] for row in rounds),
        })
    return points


def capacities(rows, budgets):
    report = []
    for (shape, config, delay), points in verified_points(rows).items():
        for budget in budgets:
            passed = [point for point in points if point["sustained"] and point["max_p99_ms"] <= budget]
            highest = max(passed, key=lambda point: point["rate"], default=None)
            higher_failed = [point for point in points if point not in passed and
                             (highest is None or point["rate"] > highest["rate"])]
            report.append({
                "shape": shape, "config": config, "delay_ms": delay, "budget_ms": budget,
                "highest_verified": highest,
                "next_failed": min(higher_failed, key=lambda point: point["rate"], default=None),
                "points": points,
            })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--budgets", nargs="+", type=float, default=[30, 50, 100])
    args = parser.parse_args()
    data = [json.loads(line) for path in args.inputs for line in path.read_text().splitlines()]
    print(json.dumps(capacities(data, args.budgets), indent=2))
