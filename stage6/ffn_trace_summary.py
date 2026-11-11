"""Attribute GPU kernel time in the Stage 6 FFN A/B Nsight Systems traces."""

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
from statistics import mean, median

ENCODER = "/sem_seg_head/transformer/encoder/layers."
GROUPS = ("norm1_input_cast", "linear1", "linear2_and_output_boundary")


def layer_groups(inspector):
    layers = json.loads(inspector.read_text())["Layers"]
    groups = {}
    for block in range(6):
        prefix = f"[ONNX Layer: {ENCODER}{block}"
        index, = [i for i, layer in enumerate(layers)
                  if f"{prefix}/linear1/MatMul]" in layer.get("Metadata", "")]
        # The tail goes through the next attention input Add, or the encoder
        # output slice/transpose for the final block. Both plans cover that
        # work, but fuse it into different layers.
        tail_end = (f"[ONNX Layer: {ENCODER}{block + 1}/Add]" if block < 5
                    else "[ONNX Layer: /sem_seg_head/Transpose]")
        end = next(i for i in range(index + 1, len(layers))
                   if tail_end in layers[i].get("Metadata", ""))
        if f"{prefix}/norm1/LayerNormalization]" not in layers[index - 1].get("Metadata", ""):
            raise ValueError(f"Unexpected FFN input boundary for encoder block {block}")
        spans = (layers[index - 1:index], layers[index:index + 1], layers[index + 1:end + 1])
        for group, selected in zip(GROUPS, spans):
            for layer in selected:
                name = layer["Name"]
                if name in groups:
                    raise ValueError(f"Duplicate FFN layer: {name}")
                groups[name] = (block, group)
    return groups


def trace_samples(database, groups, iterations):
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        ranges = db.execute("""
            SELECT n.start, n.end, n.globalTid, coalesce(n.text, s.value)
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
              WHERE n.end IS NOT NULL
            ORDER BY n.start
        """).fetchall()
        kernels = db.execute("""
            SELECT r.start, r.globalTid, k.start, k.end, k.streamId, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r USING (correlationId)
            JOIN StringIds s ON k.shortName = s.id
            ORDER BY k.start
        """).fetchall()
        count, = db.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    if len(kernels) != count or len({k[4] for k in kernels}) != 1:
        raise ValueError("Expected one CUDA launch per kernel and a single kernel stream")

    enqueues = [r for r in ranges if r[3] == "ExecutionContext::enqueueV3"]
    layer_ranges = [r for r in ranges if r[3] in groups]
    if not 0 < iterations <= len(enqueues):
        raise ValueError(f"Only {len(enqueues)} enqueue ranges in capture")
    enqueue_starts = [r[0] for r in enqueues]
    layer_starts = [r[0] for r in layer_ranges]
    samples = [[] for _ in enqueues]
    for launch, tid, start, end, _, symbol in kernels:
        enqueue = bisect_right(enqueue_starts, launch) - 1
        if enqueue < 0 or not (launch < enqueues[enqueue][1] and tid == enqueues[enqueue][2]):
            continue
        layer = bisect_right(layer_starts, launch) - 1
        name = None
        if layer >= 0 and launch < layer_ranges[layer][1] and tid == layer_ranges[layer][2]:
            name = layer_ranges[layer][3]
        samples[enqueue].append((start, end, name, symbol))
    return len(enqueues), samples[-iterations:]


def report(database, inspector, iterations, output):
    groups = layer_groups(inspector)
    enqueue_count, samples = trace_samples(database, groups, iterations)
    layer_kernel_times = {}
    rows = []
    for iteration, kernels in enumerate(samples, 1):
        if {name for _, _, name, _ in kernels if name} != groups.keys():
            raise ValueError(f"Incomplete FFN layer coverage in inference {iteration}")
        if any(previous[1] > current[0] for previous, current in zip(kernels, kernels[1:])):
            raise ValueError(f"Overlapping kernels in inference {iteration}")
        for block in range(6):
            indexes = [i for i, (_, _, name, _) in enumerate(kernels)
                       if name and groups[name][0] == block]
            if indexes[-1] - indexes[0] + 1 != len(indexes):
                raise ValueError(f"Unattributed kernel inside encoder block {block}, inference {iteration}")
        totals = Counter()
        for start, end, name, symbol in kernels:
            duration = end - start
            totals["all_kernels"] += duration
            if name:
                block, group = groups[name]
                totals[group] += duration
                totals["ffn_total"] += duration
                totals[f"block_{block}"] += duration
                layer_kernel_times.setdefault((block, group, name, symbol), []).append(duration)
        totals["other_kernels"] = totals["all_kernels"] - totals["ffn_total"]
        totals["kernel_span"] = kernels[-1][1] - kernels[0][0]
        rows.append({"iteration": iteration, **{k: v / 1e6 for k, v in totals.items()}})
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with output.with_suffix(".kernels.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("encoder_block", "group", "trt_layer", "cuda_kernel", "launches", "mean_us", "median_us"))
        for (block, group, name, symbol), durations in sorted(layer_kernel_times.items()):
            writer.writerow((block, group, name, symbol, len(durations),
                             f"{mean(durations) / 1000:.6f}", f"{median(durations) / 1000:.6f}"))
    print(f"{database.name}: {enqueue_count} captured enqueues, last {iterations} summarized")
    for key in (*GROUPS, "ffn_total", "other_kernels", "all_kernels", "kernel_span"):
        print(f"{key}: mean {mean(row[key] for row in rows):.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("inspector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    report(args.sqlite, args.inspector, args.iterations, args.output)
