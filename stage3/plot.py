"""Plot verified cumulative accuracy and same-scope Stage 3 benchmark results."""
import argparse
from html import escape
import json
from pathlib import Path


def plot(results):
    modes = [("PyTorch FP32", "pytorch", "pt-pq"), ("TensorRT FP32", "fp32", "fp32-pq-rerun")]
    data = json.loads((results / "summary.json").read_text())
    accuracy = [(name, json.loads((results / run / "result.json").read_text())["scores"]["PQ"])
                for name, _, run in modes if (results / run / "result.json").exists()]
    stage2 = results.resolve().parents[1] / "stage2/results/evaluate/1024-4/result.json"
    accuracy[:0] = [("Stage 1 official", 52.9567),
                    ("Stage 2 1024/B4", json.loads(stage2.read_text())["scores"]["PQ"])]
    panels = [("COCO val2017 PQ", "PQ", accuracy)]
    for label, unit, key in (
        ("GPU network P95", "ms", "network_ms_p95"),
        ("End-to-end P95", "ms", "e2e_ms_p95"),
        ("Offline throughput", "img/s", "images_per_second"),
    ):
        rows = [(name, data["performance"][backend][key]["median"])
                for name, backend, _ in modes if backend in data["performance"]]
        if rows:
            panels.append((label, unit, rows))

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="530" viewBox="0 0 1060 530">',
           '<rect width="1060" height="530" rx="16" fill="#f3f6fa"/>',
           '<g font-family="system-ui, sans-serif" fill="#172b43">']

    def label(x, y, content, size, weight=400):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{escape(content)}</text>')

    label(30, 39, "MaskDINO R50 panoptic · RTX 4090", 24, 700)
    label(30, 62, "Stage 3 performance · 1024×1024 · batch 1 · single slot · median of 3 rounds", 13)
    palette = {"PyTorch FP32": "#457caa", "TensorRT FP32": "#168477"}
    for index, (title, unit, rows) in enumerate(panels):
        x, y = 24 + 512 * (index % 2), 80 + 215 * (index // 2)
        svg.append(f'<rect x="{x}" y="{y}" width="500" height="202" rx="10" fill="white"/>')
        label(x + 18, y + 28, title, 17, 650)
        ceiling = max(value for _, value in rows) * 1.08
        for row, (name, value) in enumerate(rows):
            cy = y + 53 + row * 30
            length = 245 * value / ceiling
            color = palette.get(name, "#788a9e")
            label(x + 18, cy + 4, name, 12)
            svg.append(f'<rect x="{x + 152}" y="{cy - 8}" width="{length:.2f}" height="15" rx="3" fill="{color}"/>')
            label(x + 405, cy + 4, f"{value:.4f}" if unit == "PQ" else f"{value:.2f} {unit}", 12, 550)
    svg.append('</g></svg>')
    (results / "comparison.svg").write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    plot(parser.parse_args().results)
