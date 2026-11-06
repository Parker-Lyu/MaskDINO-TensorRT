"""Summarize CUDA activity within request NVTX ranges in an Nsight SQLite export."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import median

STAGES = {"decode", "preprocess", "preprocess_gpu", "H2D", "network", "postprocess", "D2H", "output_wait",
          "preprocess_wait", "network_wait", "postprocess_wait", "d2h_wait"}
TRACE_LABELS = {
    "network": "CPU / network call (GPU may continue after return)",
    "postprocess": "CPU / postprocess (includes CUDA waits)",
}
STAGE_BY_LABEL = {label: stage for stage, label in TRACE_LABELS.items()}
STAGE_BY_LABEL.update({
    "CPU / submit CUDA postprocess": "postprocess",
    "CPU / submit D2H copies": "D2H",
    "CPU / wait for preprocess": "preprocess_wait",
    "CPU / wait for network": "network_wait",
    "CPU / wait for postprocess": "postprocess_wait",
    "CPU / wait for D2H copies": "d2h_wait",
    "CPU / wait for output (network + postprocess + copies)": "output_wait",
})
ACTIVITIES = ("KERNEL", "MEMCPY", "MEMSET")


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap(intervals, start, end):
    return sum(max(0, min(end, right) - max(start, left)) for left, right in intervals if left < end and right > start)


def summarize(path):
    with sqlite3.connect(path) as db:
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ranges = db.execute("""
            SELECT coalesce(e.text, s.value), e.start, e.end
            FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
            WHERE e.end IS NOT NULL ORDER BY e.start
        """).fetchall()
        requests = [(start, end) for label, start, end in ranges if label and label.startswith("image ")]
        assert len(requests) == 32
        activities = [
            event
            for kind in ACTIVITIES
            if (table := f"CUPTI_ACTIVITY_KIND_{kind}") in tables
            for event in db.execute(f"SELECT start, end, correlationId FROM {table}")
        ]
        assert activities
        gpu = union((start, end) for start, end, _ in activities)
        api = [
            event
            for kind in ("RUNTIME", "DRIVER")
            if (table := f"CUPTI_ACTIVITY_KIND_{kind}") in tables
            for event in db.execute(f"""
                SELECT s.value, a.start, a.end, a.correlationId FROM {table} a
                JOIN StringIds s ON a.nameId = s.id
            """)
        ]

    stages = defaultdict(list)
    for label, start, end in ranges:
        label = STAGE_BY_LABEL.get(label, label)
        if label in STAGES:
            stages[label].append((start, end))
    stages["request"] = requests
    sync = defaultdict(lambda: {"count": 0, "host_ms": 0.0})
    correlation = {}
    for name, start, end, event_id in api:
        for stage, intervals in stages.items():
            if stage != "request" and any(left <= start < right for left, right in intervals):
                correlation[event_id] = stage
                if "Synchronize" in name:
                    key = f"{stage}/{name}"
                    sync[key]["count"] += 1
                    sync[key]["host_ms"] += (end - start) / 1e6
                break
    submitted = defaultdict(list)
    for start, end, event_id in activities:
        if event_id in correlation:
            submitted[correlation[event_id]].append((start, end))
    result = {"requests": len(requests), "synchronization_apis": dict(sync), "ranges": {}}
    for name, intervals in stages.items():
        host = sum(end - start for start, end in intervals)
        gpu_busy = sum(overlap(gpu, start, end) for start, end in intervals)
        result["ranges"][name] = {
            "count": len(intervals), "median_host_ms": median(end - start for start, end in intervals) / 1e6,
            "total_host_ms": host / 1e6, "overlapping_gpu_activity_ms": gpu_busy / 1e6,
            "gpu_idle_fraction": (host - gpu_busy) / host,
            "gpu_ms_attributed_to_cuda_api_calls": sum(end - start for start, end in union(submitted[name])) / 1e6,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_sqlite", type=Path)
    print(json.dumps(summarize(parser.parse_args().trace_sqlite), indent=2))
