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
            color = "#168477" if name != "B1/S1" else "#546a85"
            label(x + 16, y + 4, name, 12, 600)
            bar = (236 - label_width) * value / ceiling
            svg.append(f'<rect x="{x+label_width}" y="{y-8}" width="{bar:.2f}" height="15" rx="3" fill="{color}"/>')
            label(x + 245, y + 4, f"{value:.2f} {unit}", 12)
    svg.append("</g></svg>")
    path.write_text("\n".join(svg) + "\n")


def plot(results, finalist=None, comparison=None):
    groups = defaultdict(list)
    for row in json.loads((results / "screen/summary.json").read_text()):
        groups[row["config"]].append(row)

    # Keep the initial measurements for controls repeated in a follow-up.
    followups = defaultdict(list)
    for summary in sorted(results.glob("screen-b8*/summary.json"), key=lambda path: path.parent.name):
        for row in json.loads(summary.read_text()):
            if row["config"] not in groups:
                followups[row["config"]].append(row)

    screening = {**groups, **followups}
    labels = []
    for name in screening:
        label = name.upper().replace("S", "/S")
        labels.append(f"{label}*" if name in followups else label)
    columns = [[median(metric(row) for row in rows) for rows in screening.values()] for _, _, metric in PANELS]
    subtitle = "RTX 4090 · 1024 × 1024 · TensorRT FP32 · same 500 JPEGs · median of 3 rounds"
    screen_subtitle = subtitle
    if followups:
        screen_subtitle += " · * B8 follow-up runs"
    draw(results / "screen.svg", "MaskDINO R50 panoptic · Stage 6 offline screening",
         screen_subtitle, labels, columns, 76)
    if finalist is None:
        return
    if finalist not in groups:
        groups[finalist] = followups[finalist]
    assert len(groups[finalist]) == 3
    configs = sorted((config for config in groups if config.startswith("b1s")), key=lambda config: int(config[3:]))
    if finalist not in configs:
        configs.append(finalist)
    labels = [config.upper().replace("S", "/S") for config in configs]
    if finalist in followups:
        labels[configs.index(finalist)] += "*"
        subtitle += " · * follow-up run"
    columns = [[median(metric(row) for row in groups[config]) for config in configs] for _, _, metric in PANELS]
    draw(comparison or results / "comparison.svg", "MaskDINO R50 panoptic · Stage 5 offline slot screening",
         subtitle, labels, columns, 76)


def plot_precision(results, output):
    precision = Path(__file__).resolve().parent / "results/precision"
    sources = {
        "FP32 / TF32 off": results / "screen/summary.json",
        "FP32 / TF32 on": precision / "on/screen/summary.json",
        "FP16 mix / TF32 on": precision / "autocast-on/screen/summary.json",
    }
    labels, groups = [], []
    for policy, path in sources.items():
        rows = json.loads(path.read_text())
        for slots in range(1, 5):
            group = [row for row in rows if row["config"] == f"b1s{slots}"]
            assert sorted(row["round"] for row in group) == [1, 2, 3]
            labels.append(f"{policy} · S{slots}")
            groups.append(group)
    columns = [[median(metric(row) for row in group) for group in groups] for _, _, metric in PANELS]
    draw(output, "MaskDINO R50 panoptic · B1 precision and slot screening",
         "RTX 4090 · same 500 JPEGs per round · 3-round medians · TF32 off from the historical screen",
         labels, columns, 158)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--finalist")
    parser.add_argument("--comparison", type=Path, help="Output path for the Stage 5 slot comparison")
    parser.add_argument("--precision-comparison", type=Path, help="Output path for the three-policy slot comparison")
    args = parser.parse_args()
    if args.precision_comparison:
        plot_precision(args.results, args.precision_comparison)
    else:
        plot(args.results, args.finalist, args.comparison)
