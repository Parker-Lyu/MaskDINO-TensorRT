"""Summarize kernel overlap across CUDA streams from Nsight Systems SQLite."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def kernel_overlap(path):
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        intervals = connection.execute(
            "SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end > start"
        ).fetchall()
    if not intervals:
        raise ValueError("No kernels in the trace")
    events = sorted((timestamp, stream, direction)
                    for begin, end, stream in intervals
                    for timestamp, direction in ((begin, 1), (end, -1)))
    stream_counts = Counter()
    duration = Counter()
    previous = events[0][0]
    for timestamp, stream, direction in events:
        duration[len(stream_counts)] += timestamp - previous
        stream_counts[stream] += direction
        if stream_counts[stream] == 0:
            del stream_counts[stream]
        previous = timestamp
    span = events[-1][0] - events[0][0]
    busy = span - duration[0]
    overlap = sum(value for count, value in duration.items() if count > 1)
    return {
        "kernel_count": len(intervals),
        "kernel_streams": len({row[2] for row in intervals}),
        "kernel_window_ms": span / 1e6,
        "kernel_busy_ms": busy / 1e6,
        "multiple_kernel_streams_ms": overlap / 1e6,
        "multiple_kernel_streams_share_of_busy": overlap / busy,
        "kernel_time_ms_by_active_streams": {key: value / 1e6 for key, value in sorted(duration.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    print(json.dumps(kernel_overlap(parser.parse_args().sqlite.resolve()), indent=2))
