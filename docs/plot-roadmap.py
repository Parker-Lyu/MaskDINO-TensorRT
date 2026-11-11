"""Rebuild the README performance charts from archived stage results."""

import argparse
from html import escape
import json
from pathlib import Path
import re
import runpy
from statistics import median


DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
POLICIES = ("PyTorch", "FP32 off", "FP32 on", "FP16 mix")
COLORS = ("#61748b", "#3365a6", "#067969", "#a14c17")
ENGINE_NAMES = ("FP32 / TF32 off", "FP32 / TF32 on", "FP16 mix / TF32 on")
SLOT_COUNTS = range(1, 5)
PEAK_THROUGHPUT_SLOTS = (3, 3, 2)


def read_json(path):
    return json.loads((ROOT / path).read_text())


def draw(stage, panels):
    visible_panels = [
        (title, precision, [row for row in rows if row[0] <= stage])
        for title, precision, rows in panels
    ]
    row_height = 25 if stage >= 5 else 31
    panel_height = 60 + row_height * max(len(rows) for _, _, rows in visible_panels)
    footer = []
    if stage >= 5:
        dispatch_stages = "Stages 5–6" if stage == 6 else "Stage 5"
        footer = [
            f"Latency ends at CPU result. Timing starts at JPEG processing in Stages 2–4 and at slot dispatch in {dispatch_stages}.",
            "Stage 5 PQ: — indicates a configuration without a full COCO evaluation.",
        ]
    if stage == 6:
        footer.extend([
            "Offline throughput uses the full timed run after warmup.",
            "Stage 6: FPN FP16 passed offline acceptance. FPN + FFN FP16 is experimental (26/32 image alignment).",
        ])
    height = 112 + 2 * panel_height + (8 + 20 * len(footer) if footer else 0)
    bar_start = 250 if stage >= 5 else 226
    bar_width = 381 - bar_start
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title">',
        f'<title id="title">Stage 1–{stage} cumulative results</title>',
        f'<rect width="1000" height="{height}" rx="16" fill="#f3f6fa"/>',
        '<g font-family="system-ui, sans-serif" fill="#172b43">',
    ]

    def label(x, y, content, size=15, weight=400):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{escape(content)}</text>')

    label(28, 36, f"Stage 1–{stage} · Cumulative results", 24, 700)
    label(28, 61, "RTX 4090 · COCO val2017 PQ (5,000 images) · 500-image offline timing, 3-run medians", 14)

    for index, (title, precision, rows) in enumerate(visible_panels):
        x = 20 + 486 * (index % 2)
        y = 76 + (panel_height + 16) * (index // 2)
        svg.append(f'<rect x="{x}" y="{y}" width="474" height="{panel_height}" rx="10" fill="white"/>')
        label(x + 16, y + 29, title, 17, 700)
        ceiling = max(value for _, _, value, _ in rows if value is not None)
        for row, (number, name, value, color) in enumerate(rows):
            cy = y + 56 + row_height * row
            if stage >= 4 and row > 0 and number != rows[row - 1][0]:
                divider_y = cy - (row_height + 1) // 2
                svg.append(f'<path d="M{x + 16} {divider_y}h442" stroke="#97aabd" stroke-dasharray="4 4"/>')
            label(x + 16, cy + 5, name, 14)
            if value is not None:
                svg.append(f'<rect x="{x + bar_start}" y="{cy - 9}" width="{bar_width * value / ceiling:.2f}" height="17" rx="3" fill="{color}"/>')
            label(x + 396, cy + 5, "—" if value is None else f"{value:.{precision}f}", 14, 650)

    for i, line in enumerate(footer):
        label(28, height - 14 - 20 * (len(footer) - 1 - i), line, 14)
    svg.append("</g></svg>")
    (DOCS / f"stage{stage}-cumulative.svg").write_text("\n".join(svg) + "\n")


def draw_performance(filename, title, subtitle, panels, show_legend=False):
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="342" viewBox="0 0 1000 342" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(subtitle)}</desc>',
        '<rect width="1000" height="342" rx="16" fill="#f3f6fa"/>',
        '<g font-family="system-ui, sans-serif" fill="#172b43">',
    ]

    def label(x, y, content, size=14, weight=400, anchor="start"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                   f'text-anchor="{anchor}">{escape(content)}</text>')

    label(24, 32, title, 23, 700)
    label(24, 55, subtitle, 14)
    for index, (heading, limit, groups) in enumerate(panels):
        x = 20 + 486 * index
        svg.append(f'<rect x="{x}" y="69" width="474" height="218" rx="10" fill="white"/>')
        label(x + 16, 96, heading, 17, 700)
        for tick in range(0, limit + 1, 20):
            y = 251 - 130 * tick / limit
            svg.append(f'<path d="M{x + 42} {y:.2f}h416" stroke="#dbe3ed"/>')
            label(x + 35, f"{y + 4:.2f}", str(tick), 12, anchor="end")

        group_width = 416 / len(groups)
        for group_index, (name, values) in enumerate(groups):
            center = x + 42 + group_width * (group_index + 0.5)
            label(center, 274, name, 13, 650, "middle")
            for bar_index, (policy, value) in enumerate(values):
                bar_x = center + 34 * (bar_index - (len(values) - 1) / 2)
                bar_height = 130 * value / limit
                bar_y = 251 - bar_height
                svg.append(f'<rect x="{bar_x - 12:.2f}" y="{bar_y:.2f}" width="24" '
                           f'height="{bar_height:.2f}" rx="3" fill="{COLORS[policy]}"/>')
                label(f"{bar_x:.2f}", f"{bar_y - 5:.2f}", f"{value:.2f}",
                      11 if len(values) > 1 else 12, 650, "middle")

    if show_legend:
        for i, (name, color) in enumerate(zip(("PyTorch FP32", *ENGINE_NAMES), COLORS)):
            x = 152 + 180 * i
            svg.append(f'<rect x="{x}" y="309" width="12" height="12" rx="3" fill="{color}"/>')
            label(x + 18, 320, name, 13)
    else:
        label(500, 320, "TensorRT throughput: FP32 off S3 · FP32 on S3 · FP16 mix S2", 14, anchor="middle")
    svg.append("</g></svg>")
    (DOCS / filename).write_text("\n".join(svg) + "\n")


def draw_precision(latency, throughput):
    latency_heading = "Single-slot end-to-end P95 (ms) ↓"
    throughput_heading = "Offline throughput (img/s) ↑"

    def overview(baseline, values):
        return [(name, [(i, value)]) for i, (name, value) in enumerate(zip(POLICIES, [baseline, *values]))]

    draw_performance(
        "precision-performance.svg", "PyTorch vs. TensorRT",
        "RTX 4090 · 1024 × 1024 · batch 1 · 500 JPEGs · medians of three runs",
        (
            (latency_heading, 60, overview(latency[0][0], latency[2])),
            (throughput_heading, 80, overview(throughput[0][0], throughput[3])),
        ),
    )
    draw_performance(
        "roadmap-performance.svg", "Inference performance by stage",
        "Stage 2 · PyTorch     Stage 3 · TensorRT network     Stage 4 · Native C++/CUDA     Stage 5 · Multi-slot",
        (
            (latency_heading, 60, roadmap_groups(latency[:3])),
            (throughput_heading, 80, roadmap_groups(throughput)),
        ),
        show_legend=True,
    )


def roadmap_groups(values_by_stage):
    return [
        (f"Stage {stage}", list(enumerate(values, 0 if stage == 2 else 1)))
        for stage, values in enumerate(values_by_stage, 2)
    ]


def engine_rows(values_by_stage):
    return [
        (stage, f"Stage {stage} · {name}", value, COLORS[i + 1])
        for stage, values in enumerate(values_by_stage, 3)
        for i, (name, value) in enumerate(zip(ENGINE_NAMES, values))
    ]


def slot_rows(values_by_engine):
    return [
        (5, f"Stage 5 · {name} · S{slots}", value, COLORS[i + 1])
        for i, (name, values) in enumerate(zip(ENGINE_NAMES, values_by_engine))
        for slots, value in zip(SLOT_COUNTS, values)
    ]


def stage6_panels(throughput):
    fpn_path = "stage6/results/fpn-half-ab"
    fpn = read_json(f"{fpn_path}/summary.json")["native"]["fpn-half"]
    ffn = read_json("stage6/results/ffn-full-half/summary.json")
    rows = {key: [] for key in ("pq", "network_p95_ms", "end_to_end_p95_ms", "makespan_images_per_second")}
    for slots in (1, 2):
        config = f"b1s{slots}"
        fpn_pq = read_json(f"{fpn_path}/pq/{config}/validation.json")["scores"]["PQ"]
        for engine, name, color, timing, pq in (
            ("fpn-half", "FPN FP16", "#348f79", fpn[config], fpn_pq),
            ("ffn-half", "FPN + FFN", "#8051a3", ffn["native"]["ffn-half"][config], ffn["pq"]["scores"]["PQ"]),
        ):
            label = f"Stage 6 · {name} · B1/S{slots}"
            values = {**timing, "pq": pq, "makespan_images_per_second": throughput[engine, slots]}
            for key, panel_rows in rows.items():
                panel_rows.append((6, label, values[key], color))
    return rows


def main(cumulative_only=False):
    official_log = ROOT / "stage1/results/official-r50-20261106-6eb01c1.log"
    official_pq = float(re.search(r"copypaste: (\d+\.\d+),", official_log.read_text())[1])
    pq_paths = (
        "stage2/results/evaluate/1024-4/result.json",
        "stage3/results/fp32-pq-rerun/result.json",
        "stage4/results/native-pq/result.json",
    )
    square_pq, python_off_pq, native_off_pq = (read_json(path)["scores"]["PQ"] for path in pq_paths)

    stage2 = next(row["statistics"] for row in read_json("stage2/results/benchmark/summary.json")
                  if (row["size"], row["batch"]) == (1024, 1))
    stage3 = read_json("stage3/results/summary.json")["performance"]["fp32"]
    precision = "stage3/results/precision"
    python_engines = (
        stage3,
        read_json(f"{precision}/benchmark/on-summary.json")["performance"]["fp32"],
        read_json(f"{precision}/benchmark/autocast-on-summary.json")["performance"]["mixed"],
    )
    tf32_pair = read_json("stage4/results/precision/tf32-pair/summary.json")
    autocast_pair = read_json("stage4/results/precision/autocast-pair/summary.json")
    native_engines = (
        read_json("stage4/results/summary.json")["performance"]["native"],
        tf32_pair["performance"]["on"],
        autocast_pair["performance"]["mixed"],
    )
    screen_paths = (
        "stage5/results/screen/summary.json",
        "stage5/results/precision/on/screen/summary.json",
        "stage5/results/precision/autocast-on/screen/summary.json",
    )
    slot_runs = []
    for path in screen_paths:
        screen = read_json(path)
        groups = [[row for row in screen if row["config"] == f"b1s{slots}"] for slots in SLOT_COUNTS]
        assert all(sorted(row["round"] for row in runs) == [1, 2, 3] for runs in groups), path
        slot_runs.append(groups)

    def performance(stage2_key, key, slot_metric):
        return (
            [stage2[stage2_key]["median"]],
            [engine[key]["median"] for engine in python_engines],
            [engine[key]["median"] for engine in native_engines],
            [[median(slot_metric(row) for row in runs) for runs in groups] for groups in slot_runs],
        )

    throughput = performance("images_per_s", "images_per_second", lambda row: row["makespan_images_per_second"])
    latency = performance("e2e_p95_ms", "e2e_ms_p95", lambda row: row["latency_ms"]["p95"])
    network = performance("network_p95_ms", "network_ms_p95", lambda row: row["network_ms_per_batch"]["p95"])
    python_pq = [python_off_pq] + [
        read_json(f"{precision}/precision-pq-{variant}/result.json")["scores"]["PQ"]
        for variant in ("on", "autocast-on")
    ]
    native_pq = [native_off_pq, tf32_pair["accuracy"]["scores"]["PQ"], autocast_pair["accuracy"]["scores"]["PQ"]]
    slots_off_pq = [
        read_json(f"stage5/results/pq/b1s{slots}/validation.json")["scores"]["PQ"]
        for slots in (1, 2, 3)
    ]
    # Full COCO results cover only B1/S1–S3 with TF32 off.
    unavailable = [None] * len(SLOT_COUNTS)
    overviews = runpy.run_path(str(DOCS / "plot-stage6-overviews.py"))
    stage6_throughput = overviews["stage6_throughput"]()
    stage6 = stage6_panels(stage6_throughput)
    pq = [
        (1, "Stage 1 · Official input", official_pq, COLORS[0]),
        (2, "Stage 2 · 1024/B4", square_pq, COLORS[0]),
        *engine_rows((python_pq, native_pq)),
        *slot_rows(([*slots_off_pq, None], unavailable, unavailable)),
        *stage6["pq"],
    ]
    panels = [("PQ on full COCO val2017", 4, pq)]
    for title, values, stage6_key in (
        ("Network GPU P95 (ms/batch)", network, "network_p95_ms"),
        ("End-to-end P95 (ms)", latency, "end_to_end_p95_ms"),
        ("Offline throughput (img/s)", throughput, "makespan_images_per_second"),
    ):
        baseline = (2, "Stage 2 · PyTorch", values[0][0], COLORS[0])
        panels.append((title, 2, [baseline, *engine_rows(values[1:3]), *slot_rows(values[3]), *stage6[stage6_key]]))

    for stage in (3, 4, 5, 6):
        draw(stage, panels)
    if not cumulative_only:
        overviews["main"](stage6_throughput)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cumulative-only", action="store_true", help="Rebuild only the cumulative stage charts.")
    main(cumulative_only=parser.parse_args().cumulative_only)
