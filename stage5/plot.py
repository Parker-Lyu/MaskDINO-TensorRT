"""Draw an offline comparison from the three-round Stage 5 screen."""
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
from statistics import median


def load_rows(results):
    groups = defaultdict(list)
    for row in json.loads((results / "screen/summary.json").read_text()):
        groups[row["config"]].append(row)
    return groups


def plot(results):
    groups = load_rows(results)
    panels = (
        ("Throughput", "img/s", lambda row: row["makespan_images_per_second"]),
        ("Per-image dispatch-to-result P95", "ms", lambda row: row["latency_ms"]["p95"]),
        ("Peak process VRAM", "GiB", lambda row: row["sampled_peak_process_used_bytes"] / 2**30),
    )
    height = 135 + 35 * len(groups)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}">',
        f'<rect width="1080" height="{height}" rx="16" fill="#f3f6fa"/>',
        '<g font-family="system-ui,sans-serif" fill="#172b43">',
    ]

    def label(x, y, text, size, weight=400):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{escape(text)}</text>')

    label(28, 37, "MaskDINO R50 panoptic · Stage 5 offline screening", 23, 700)
    label(28, 60, "RTX 4090 · 1024 × 1024 · TensorRT FP32 · same 500 JPEGs · median of 3 rounds", 13)
    for column, (title, unit, metric) in enumerate(panels):
        x = 24 + column * 346
        svg.append(f'<rect x="{x}" y="78" width="337" height="{height - 96}" rx="10" fill="white"/>')
        label(x + 16, 107, title, 16, 650)
        values = {name: median(metric(row) for row in rows) for name, rows in groups.items()}
        ceiling = max(values.values())
        for i, (name, value) in enumerate(values.items()):
            y = 133 + i * 35
            color = "#546a85" if name == "b1s1" else "#168477"
            label(x + 16, y + 4, name.upper(), 12, 600)
            svg.append(f'<rect x="{x+76}" y="{y-8}" width="{160*value/ceiling:.2f}" height="15" rx="3" fill="{color}"/>')
            label(x + 245, y + 4, f"{value:.2f} {unit}", 12)
    svg.append("</g></svg>")
    (results / "screen.svg").write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    plot(parser.parse_args().results)
