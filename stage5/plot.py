"""Draw comparable offline results measured on the same 500 JPEGs."""
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
from statistics import median

PANELS = (
    ("Throughput", "img/s", lambda row: row["makespan_images_per_second"]),
    ("Per-image dispatch-to-result P95", "ms", lambda row: row["latency_ms"]["p95"]),
    ("Peak process VRAM", "GiB", lambda row: row["sampled_peak_process_used_bytes"] / 2**30),
)


def draw(path, title, subtitle, labels, columns, label_width):
    height = 135 + 35 * len(labels)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}">',
        f'<rect width="1080" height="{height}" rx="16" fill="#f3f6fa"/>',
        '<g font-family="system-ui,sans-serif" fill="#172b43">',
    ]

    def label(x, y, content, size, weight=400):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{escape(content)}</text>')

    label(28, 37, title, 23, 700)
    label(28, 60, subtitle, 13)
    for column, ((title, unit, _), values) in enumerate(zip(PANELS, columns)):
        x = 24 + column * 346
        svg.append(f'<rect x="{x}" y="78" width="337" height="{height - 96}" rx="10" fill="white"/>')
        label(x + 16, 107, title, 16, 650)
        ceiling = max(values)
        for i, (name, value) in enumerate(zip(labels, values)):
            y = 133 + i * 35
            color = "#168477" if name.startswith("Stage 5") or (label_width == 76 and name != "B1S1") else "#546a85"
            label(x + 16, y + 4, name, 12, 600)
            bar = (236 - label_width) * value / ceiling
            svg.append(f'<rect x="{x+label_width}" y="{y-8}" width="{bar:.2f}" height="15" rx="3" fill="{color}"/>')
            label(x + 245, y + 4, f"{value:.2f} {unit}", 12)
    svg.append("</g></svg>")
    path.write_text("\n".join(svg) + "\n")


def plot(results, finalist=None):
    groups = defaultdict(list)
    for row in json.loads((results / "screen/summary.json").read_text()):
        groups[row["config"]].append(row)
    columns = [[median(metric(row) for row in rows) for rows in groups.values()] for _, _, metric in PANELS]
    subtitle = "RTX 4090 · 1024 × 1024 · FP32 · same 500 JPEGs · median of 3 rounds"
    draw(results / "screen.svg", "MaskDINO R50 panoptic · Stage 5 offline screening",
         subtitle.replace("FP32", "TensorRT FP32"), [name.upper() for name in groups], columns, 76)
    if finalist is None:
        return
    if finalist not in groups:
        for run in results.glob("screen-b8*/summary.json"):
            for row in json.loads(run.read_text()):
                if row["config"] == finalist:
                    groups[finalist].append(row)
    assert len(groups[finalist]) == 3
    stage4 = results.resolve().parents[1] / "stage4/results/summary.json"
    previous = json.loads(stage4.read_text())["performance"]
    cumulative = []
    for (_, _, metric), name, scale in zip(PANELS, ("images_per_second", "e2e_ms_p95",
                                                   "sampled_peak_process_used_bytes"), (1, 1, 2**30)):
        cumulative.append([previous[key][name]["median"] / scale for key in ("pytorch", "reference", "native")]
                          + [median(metric(row) for row in groups[config]) for config in ("b1s1", finalist)])
    formatted = finalist.upper().replace("S", "/S")
    labels = ["PyTorch", "TRT + Python", "Stage 4 C++", "Stage 5 B1/S1", f"Stage 5 {formatted}"]
    draw(results / "comparison.svg", "MaskDINO R50 panoptic · offline comparison", subtitle, labels, cumulative, 127)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--finalist")
    args = parser.parse_args()
    plot(args.results, args.finalist)
