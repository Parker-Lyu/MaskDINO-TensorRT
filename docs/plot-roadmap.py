"""Rebuild the README performance charts from archived stage results."""

from html import escape
import json
from pathlib import Path
import re
from statistics import median


DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
POLICIES = ("PyTorch", "FP32 off", "FP32 on", "FP16 mix")
COLORS = ("#61748b", "#3365a6", "#067969", "#a14c17")


def read_json(path):
    return json.loads((ROOT / path).read_text())


def draw(stage, panels):
    panel_height = 60 + 31 * stage
    height = 112 + 2 * panel_height + (28 if stage == 5 else 0)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title">',
        f'<title id="title">Stage 1–{stage} cumulative results</title>',
        f'<rect width="1000" height="{height}" rx="16" fill="#f3f6fa"/>',
        '<g font-family="system-ui, sans-serif" fill="#172b43">',
    ]

    def label(x, y, content, size=15, weight=400):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{escape(content)}</text>')

    label(28, 36, f"Stage 1–{stage} · Cumulative results", 24, 700)
    label(28, 61, "RTX 4090 · COCO val2017 PQ · 500-image offline tests, median of three runs per stage", 14)

    for index, (title, precision, rows) in enumerate(panels):
        x = 20 + 486 * (index % 2)
        y = 76 + (panel_height + 16) * (index // 2)
        svg.append(f'<rect x="{x}" y="{y}" width="474" height="{panel_height}" rx="10" fill="white"/>')
        label(x + 16, y + 29, title, 17, 700)
        ceiling = max(value for _, _, value in rows)
        for row, (number, name, value) in enumerate(item for item in rows if item[0] <= stage):
            cy = y + 56 + 31 * row
            color = "#168477" if number == stage else "#54769a"
            label(x + 16, cy + 5, name, 14)
            svg.append(f'<rect x="{x + 170}" y="{cy - 9}" width="{211 * value / ceiling:.2f}" height="17" rx="3" fill="{color}"/>')
            label(x + 396, cy + 5, f"{value:.{precision}f}", 14, 650)

    if stage == 5:
        label(28, height - 14,
              "Latency ends at CPU result. Timing starts at JPEG processing in Stages 2–4 and at slot dispatch in Stage 5.", 14)
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
                label(f"{bar_x:.2f}", f"{bar_y - 5:.2f}", f"{value:.2f}", 12, 650, "middle")

    if show_legend:
        for i, (name, color) in enumerate(zip(
                ("PyTorch FP32", "FP32 / TF32 off", "FP32 / TF32 on", "FP16 mix / TF32 on"), COLORS)):
            x = 152 + 180 * i
            svg.append(f'<rect x="{x}" y="309" width="12" height="12" rx="3" fill="{color}"/>')
            label(x + 18, 320, name, 13)
    else:
        label(500, 320, "TensorRT throughput: FP32 off S3 · FP32 on S3 · FP16 mix S2", 14, anchor="middle")
    svg.append("</g></svg>")
    (DOCS / filename).write_text("\n".join(svg) + "\n")


def draw_precision(latency, throughput):
    native = (
        read_json("stage4/results/precision/tf32-pair/summary.json")["performance"]["on"],
        read_json("stage4/results/precision/autocast-pair/summary.json")["performance"]["mixed"],
    )
    native_latency = [latency[2][2], *(row["e2e_ms_p95"]["median"] for row in native)]
    native_throughput = [throughput[2][2], *(row["images_per_second"]["median"] for row in native)]
    # Stage 5 screening medians published in stage5/README.md, using S3 / S3 / S2.
    slot_throughput = [throughput[3][2], 66.43, 74.35]
    latency_heading = "Single-slot end-to-end P95 (ms) ↓"
    throughput_heading = "Offline throughput (img/s) ↑"

    def overview(baseline, values):
        return [(name, [(i, value)]) for i, (name, value) in enumerate(zip(POLICIES, [baseline, *values]))]

    draw_performance(
        "precision-performance.svg", "PyTorch vs. TensorRT",
        "RTX 4090 · 1024 × 1024 · batch 1 · 500 JPEGs · medians of three runs",
        (
            (latency_heading, 60, overview(latency[0][2], native_latency)),
            (throughput_heading, 80, overview(throughput[0][2], slot_throughput)),
        ),
    )
    draw_performance(
        "roadmap-performance.svg", "Inference performance by stage",
        "Stage 2 · PyTorch     Stage 3 · TensorRT network     Stage 4 · Native C++/CUDA     Stage 5 · Multi-slot",
        (
            (latency_heading, 60, [
                ("Stage 2", [(0, latency[0][2])]),
                ("Stage 3", [(1, latency[1][2])]),
                ("Stage 4", list(enumerate(native_latency, 1))),
            ]),
            (throughput_heading, 80, [
                ("Stage 2", [(0, throughput[0][2])]),
                ("Stage 3", [(1, throughput[1][2])]),
                ("Stage 4", list(enumerate(native_throughput, 1))),
                ("Stage 5", list(enumerate(slot_throughput, 1))),
            ]),
        ),
        show_legend=True,
    )


def main():
    official_log = ROOT / "stage1/results/official-r50-20261106-6eb01c1.log"
    official_pq = float(re.search(r"copypaste: (\d+\.\d+),", official_log.read_text())[1])
    pq_paths = (
        "stage2/results/evaluate/1024-4/result.json",
        "stage3/results/fp32-pq-rerun/result.json",
        "stage4/results/native-pq/result.json",
        "stage5/results/pq/b1s3/validation.json",
    )
    pq_names = ("Stage 1 · Official input", "Stage 2 · 1024/B4", "Stage 3 · B1/S1", "Stage 4 · B1/S1", "Stage 5 · B1/S3")
    pq_values = [official_pq] + [read_json(path)["scores"]["PQ"] for path in pq_paths]
    pq = [(number, name, value) for number, (name, value) in enumerate(zip(pq_names, pq_values), 1)]

    stage2 = next(row["statistics"] for row in read_json("stage2/results/benchmark/summary.json")
                  if (row["size"], row["batch"]) == (1024, 1))
    stage3 = read_json("stage3/results/summary.json")["performance"]["fp32"]
    stage4 = read_json("stage4/results/summary.json")["performance"]["native"]
    stage5 = [row for row in read_json("stage6/results/screen/summary.json") if row["config"] == "b1s3"]

    def performance_rows(stage2_key, key, stage5_values):
        return [
            (2, "Stage 2 · PyTorch", stage2[stage2_key]["median"]),
            (3, "Stage 3 · Python/TRT", stage3[key]["median"]),
            (4, "Stage 4 · C++/TRT", stage4[key]["median"]),
            (5, "Stage 5 · B1/S3", median(stage5_values)),
        ]

    throughput = performance_rows("images_per_s", "images_per_second",
                                  (row["makespan_images_per_second"] for row in stage5))
    latency = performance_rows("e2e_p95_ms", "e2e_ms_p95",
                               (row["latency_ms"]["p95"] for row in stage5))
    network = performance_rows("network_p95_ms", "network_ms_p95",
                               (row["network_ms_per_batch"]["p95"] for row in stage5))
    panels = (
        ("PQ on full COCO val2017", 4, pq),
        ("Offline throughput (img/s)", 2, throughput),
        ("Per-image latency P95 (ms)", 2, latency),
        ("Network GPU P95 (ms/batch)", 2, network),
    )
    for stage in (3, 4, 5):
        draw(stage, panels)
    draw_precision(latency, throughput)


if __name__ == "__main__":
    main()
