"""Create reproducible arrivals and summarize native per-request measurements."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import statistics


def arrivals(path, rate, seconds, shape, seed):
    randomizer = random.Random(seed)
    progress = 0.0
    times = []
    while True:
        progress += (1.0 if shape == "uniform" else randomizer.expovariate(1.0)) / rate
        elapsed = progress
        if shape == "burst":
            cycles, offset = divmod(progress, 10)
            elapsed = cycles * 10 + (offset / 2 if offset < 2 else 1 + (offset - 2) * 9 / 8)
        if elapsed >= seconds:
            break
        times.append(elapsed)
    path.write_text("".join(f"{100 + value * 1000:.9f}\n" for value in times))


def percentiles(values, levels):
    values = sorted(values)
    if not values:
        return {f"p{level}": None for level in levels}
    return {f"p{level}": values[max(0, math.ceil(len(values) * level / 100) - 1)] for level in levels}


def summarize(output, seconds=None):
    complete, rejected, makespan, memory = (output / "run.tsv").read_text().split()
    complete, rejected, makespan = int(complete), int(rejected), float(makespan)
    with (output / "requests.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == complete + rejected
    assert [int(row["request"]) for row in rows] == list(range(len(rows)))
    ok = [row for row in rows if row["status"] == "ok"]
    assert len(ok) == complete and sum(row["status"] == "rejected" for row in rows) == rejected
    batch = {int(row["group"]): row for row in ok}
    delays = [float(row["ready_ms"]) - float(row["arrival_ms"]) for row in ok]
    queue_ms = [float(row["dispatch_ms"]) - float(row["queued_ms"]) for row in ok]
    batch_wait = [float(row["batch_wait_ms"]) for row in ok]
    report = {
        "completed": complete, "rejected": rejected, "makespan_seconds": makespan,
        "makespan_images_per_second": complete / makespan,
        "sampled_peak_process_used_bytes": int(memory),
        "network_ms_per_batch": percentiles((float(row["network_ms"]) for row in batch.values()), (50, 95)),
        "latency_ms": percentiles(delays, (50, 95, 99)),
        "queue_including_batch_wait_ms": percentiles(queue_ms, (50, 95, 99)),
        "batch_wait_ms": percentiles(batch_wait, (50, 95, 99)),
        "queue_excluding_batch_wait_ms": percentiles((q - b for q, b in zip(queue_ms, batch_wait)), (50, 95, 99)),
        "batch_dispatch_histogram": dict(Counter(int(row["batch"]) for row in batch.values())),
        "queue_depth_max_at_arrival": max(int(row["arrival_depth"]) for row in rows),
    }
    if seconds is not None:
        start_ms, end_ms = 100, 100 + 1000 * seconds
        completed_in_window = sum(float(row["ready_ms"]) <= end_ms for row in ok)
        intervals = 5 if seconds < 30 else int(seconds // 10)
        window = 1000 * seconds / intervals
        pending = []
        for index in range(1, intervals + 1):
            edge = start_ms + index * window
            pending.append(sum(float(row["arrival_ms"]) <= edge < float(row["ready_ms"]) for row in ok))
        midpoint = len(pending) // 2
        last_minus_first = statistics.mean(pending[midpoint:]) - statistics.mean(pending[:midpoint])
        report.update({
            "arrival_window_seconds": seconds, "observed_arrivals_per_second": len(rows) / seconds,
            "completed_per_second_during_arrivals": completed_in_window / seconds,
            "pending_at_window_edges": pending, "mean_pending_increase": last_minus_first,
            "drain_ms": max(0, makespan * 1000 - end_ms),
            "arrival_replay_lag_ms": percentiles(
                (float(row["queued_ms"]) - float(row["arrival_ms"]) for row in rows), (50, 95, 99)
            ),
            "sustained": rejected == 0 and last_minus_first <= 2 and
            completed_in_window >= len(rows) - max(2, math.ceil(len(rows) * 0.005)),
        })
    else:
        # Trim the first and last 5% of the makespan for the steady-state estimate.
        left, right = 50 * makespan, 950 * makespan
        middle = sum(left <= float(row["ready_ms"]) < right for row in ok)
        report["steady_images_per_second"] = middle / (0.9 * makespan)
        report["steady_window_ms"] = [left, right]
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("arrivals")
    generate.add_argument("file", type=Path)
    generate.add_argument("rate", type=float)
    generate.add_argument("seconds", type=float)
    generate.add_argument("shape", choices=("uniform", "poisson", "burst"))
    generate.add_argument("seed", type=int)
    report = commands.add_parser("report")
    report.add_argument("output", type=Path)
    report.add_argument("--arrival-window", type=float)
    args = parser.parse_args()
    if args.command == "arrivals":
        arrivals(args.file, args.rate, args.seconds, args.shape, args.seed)
    else:
        print(json.dumps(summarize(args.output, args.arrival_window)))


if __name__ == "__main__":
    main()
