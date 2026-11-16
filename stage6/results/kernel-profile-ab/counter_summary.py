"""Check Nsight Compute launch identity against the trace and summarize counters."""

import argparse
import csv
from pathlib import Path
from statistics import median

METRICS = {
    "ncu_duration_ns": "gpu__time_duration.sum",
    "sm_clock_hz": "gpc__cycles_elapsed.avg.per_second",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_pipe_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "fma_pipe_pct": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_lookup_bytes": "lts__t_bytes.sum",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "register_limit_ctas": "launch__occupancy_limit_registers",
    "smem_limit_ctas": "launch__occupancy_limit_shared_mem",
    "warp_limit_ctas": "launch__occupancy_limit_warps",
    "active_warps_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "issue_active_pct": "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "eligible_warps_per_scheduler": "smsp__warps_eligible.avg.per_cycle_active",
    "long_scoreboard_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "short_scoreboard_pct": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "math_pipe_throttle_pct": "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
}
SAMPLES = 3


def plan_summary(directory, plan):
    with (directory / f"{plan}.trace.csv").open(newline="") as file:
        trace = {(row["trt_layer"], row["cuda_kernel"]): row for row in csv.DictReader(file)}
    targets = [tuple(line.split("\t")) for line in
               (directory / f"{plan}.targets.txt").read_text().splitlines()]
    with (directory / f"{plan}.ncu.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        units = next(reader)
        rows = list(reader)
    if len(rows) != SAMPLES * len(targets):
        raise ValueError(f"Unexpected launch count for {plan}: {len(rows)}")
    for output_name, unit in (("dram_read_bytes", "byte"), ("dram_write_bytes", "byte"),
                              ("l2_lookup_bytes", "byte"), ("ncu_duration_ns", "ns")):
        if units[METRICS[output_name]] != unit:
            raise ValueError("Export Nsight Compute CSV with --print-units base")

    for index, target in enumerate(targets):
        reference = trace[target]
        launches = rows[index::len(targets)]
        for launch in launches:
            identity = (launch["Kernel Name"], launch["Grid Size"], launch["Block Size"])
            if identity != (target[1], reference["grid"], reference["block"]):
                raise ValueError(f"{plan} {target[0]}: launch identity differs from trace: {identity}")
        counters = {key: median(float(row[metric]) for row in launches)
                    for key, metric in METRICS.items()}
        dram_total = [float(row[METRICS["dram_read_bytes"]])
                      + float(row[METRICS["dram_write_bytes"]]) for row in launches]
        yield {"plan": plan, **reference, "ncu_samples": SAMPLES,
               "dram_total_bytes": median(dram_total), **counters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = [row for plan in ("control", "fpn", "ffn") for row in plan_summary(args.directory, plan)]
    with (args.directory / "counters.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows) * SAMPLES} Nsight Compute launches against Nsight Systems")


if __name__ == "__main__":
    main()
