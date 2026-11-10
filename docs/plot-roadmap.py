"""Rebuild the cumulative README charts from the archived stage results."""

from html import escape
import json
from pathlib import Path
import re
from statistics import median


DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent


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


if __name__ == "__main__":
    main()
