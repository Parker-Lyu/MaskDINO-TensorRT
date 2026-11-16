"""Export FPN timings and identify the launches profiled with Nsight Compute."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
from statistics import mean
import sys

STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE))
from ffn_trace_summary import layer_groups, trace_samples

FPN = STAGE / "results/fpn-half-ab"
FFN = STAGE / "results/ffn-full-half"
CONV = "/sem_seg_head/layer_1/Conv"
ITERATIONS = 40


def fpn_groups(inspector, profile):
    layers = json.loads(inspector.read_text())["Layers"]
    groups = {}
    for name, duration in profile["layer_mean_ms"].items():
        if duration == 0:
            continue
        if name == CONV:
            group = "fpn_conv"
        elif name.startswith("__myl_Add_"):
            group = "fpn_add"
        elif name.startswith("__myl_Cast_"):
            group = "fpn_input_cast"
        else:
            group = "fpn_groupnorm_relu"
        groups[name] = group

    conv = next(layer for layer in layers if layer["Name"] == CONV)
    output_name = conv["Outputs"][0]["Name"]
    for layer in layers:
        if layer.get("LayerType") == "Reformat" and any(
            tensor["Name"] == output_name for tensor in layer["Inputs"]
        ):
            groups[layer["Name"]] = "fpn_output_reformat"
    return groups


def trace_plan(database, inspector, fpn, output):
    ffn = {name: group for name, (block, group) in layer_groups(inspector).items()
           if block == 1} if database.stem != "control" else {}
    groups = {**fpn, **ffn}
    profiled = {name for name in groups if not name.startswith("__myl_Move_")
                and groups[name] != "fpn_add"}
    enqueue_count, samples = trace_samples(database, groups, ITERATIONS)
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        launch_geometry = dict((start, (f"({gx}, {gy}, {gz})", f"({bx}, {by}, {bz})"))
                               for start, gx, gy, gz, bx, by, bz in db.execute("""
            SELECT start, gridX, gridY, gridZ, blockX, blockY, blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """))

    records = defaultdict(list)
    totals = []
    for iteration, kernels in enumerate(samples, 1):
        seen = []
        if {name for _, _, name, _ in kernels if name} != groups.keys():
            raise ValueError(f"Incomplete layer coverage for {database.name} inference {iteration}")
        timing = Counter()
        for start, end, name, symbol in kernels:
            elapsed_us = (end - start) / 1000
            timing["network_kernels"] += elapsed_us
            if name is not None:
                group = groups[name]
                grid, block = launch_geometry[start]
                records[group, name, symbol, grid, block].append(elapsed_us)
                timing[group] += elapsed_us
                if name in fpn:
                    timing["fpn_total"] += elapsed_us
                if name in profiled:
                    seen.append((name, symbol))
        if iteration == 1:
            expected = seen
        elif seen != expected:
            raise ValueError(f"Launch sequence changed in {database.name} inference {iteration}")
        totals.append({"iteration": iteration, **timing})

    with output.with_suffix(".targets.txt").open("w") as file:
        for name, symbol in expected:
            print(name, symbol, sep="\t", file=file)
    with output.with_suffix(".trace.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("scope", "trt_layer", "cuda_kernel", "grid", "block", "samples", "mean_us"))
        for key, durations in records.items():
            writer.writerow((*key, len(durations), f"{mean(durations):.3f}"))
    with output.with_suffix(".per-inference.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=totals[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.3f}" if k != "iteration" else v for k, v in row.items()}
                         for row in totals)
    print(f"{database.stem}: {enqueue_count} captured enqueues, last {ITERATIONS} summarized")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    profile = json.loads((FPN / "summary.json").read_text())["profile"]
    for label, name in (("control", "control"), ("fpn", "fpn-half"), ("ffn", None)):
        inspector = FPN / f"cached/{name}/layers.json" if name else FFN / "build/layers.json"
        groups = fpn_groups(inspector, profile[name]) if name else {}
        trace_plan(args.trace_directory / f"{label}.sqlite", inspector, groups,
                   args.output_directory / label)


if __name__ == "__main__":
    main()
