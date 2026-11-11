"""Render the two top-level README comparison charts, including Stage 6.

Stage 2–5 historical numbers match archived README/Stage 2–5 reports.
Stage 6 values come from archived FPN and FFN A/B summary JSON.
These are cross-stage milestone comparisons, not paired experiments.
"""
from html import escape
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

BLUE = "#3365a6"
TEAL = "#067969"
ORANGE = "#a14c17"
GRAY = "#61748b"
FPN = "#348f79"
FFN = "#8051a3"


def stage6_throughput():
    throughput = {}
    for path, engine in (("fpn-half-ab/cached/ab", "fpn-half"), ("ffn-full-half/ab", "ffn-half")):
        rows = json.loads((ROOT / "stage6/results" / path / "summary.json").read_text())
        for slots in (1, 2):
            runs = [row for row in rows if row["engine"] == engine and row["slots"] == slots]
            assert sorted(row["round"] for row in runs) == [1, 2, 3], (path, slots)
            throughput[engine, slots] = median(row["makespan_images_per_second"] for row in runs)
    return throughput


def stage6(throughput):
    fpn = json.loads((ROOT / "stage6/results/fpn-half-ab/summary.json").read_text())
    ffn = json.loads((ROOT / "stage6/results/ffn-full-half/summary.json").read_text())
    return {
        "fpn_p95": fpn["native"]["fpn-half"]["b1s1"]["end_to_end_p95_ms"],
        "fpn_throughput": throughput["fpn-half", 2],
        "ffn_p95": ffn["native"]["ffn-half"]["b1s1"]["end_to_end_p95_ms"],
        "ffn_throughput": throughput["ffn-half", 2],
    }


def render(path, title, subtitle, groups, foot1, foot2):
    width, height = 1260, 485
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(subtitle)}. Historical figures have different measurement boundaries; consult README for precision acceptance and comparisons.</desc>',
        f'<rect width="{width}" height="{height}" rx="16" fill="#f3f6fa"/>',
        '<g font-family="system-ui, sans-serif" fill="#172b43">',
        f'<text x="25" y="38" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="25" y="63" font-size="14">{escape(subtitle)}</text>',
    ]
    for panel in range(2):
        x = 20 + panel * 620
        key = "lat" if panel == 0 else "thr"
        maximum = 60 if panel == 0 else 100
        heading = "Single-slot end-to-end P95 (ms) ↓" if panel == 0 else "Offline throughput (images/s) ↑"
        svg.extend([
            f'<rect x="{x}" y="83" width="600" height="305" rx="10" fill="white"/>',
            f'<text x="{x + 20}" y="113" font-size="18" font-weight="700">{heading}</text>',
        ])
        for tick in range(0, maximum + 1, 20):
            y = 327 - 175 * tick / maximum
            svg.append(f'<path d="M{x + 52} {y:.2f}h530" stroke="#e2e9f1"/>')
            svg.append(f'<text x="{x + 44}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{tick}</text>')

        group_width = 530 / len(groups)
        for index, group in enumerate(groups):
            center = x + 52 + (index + 0.5) * group_width
            svg.append(f'<text x="{center:.2f}" y="351" text-anchor="middle" font-size="14" font-weight="650">{escape(group["name"])}</text>')
            active = [item for item in group["values"] if item.get(key) is not None]
            bar_width = min(30, (group_width - 12) / max(1, len(active)) - 7)
            spacing_limit = group.get("throughput_spacing", 37) if key == "thr" else 37
            spacing = min(spacing_limit, (group_width - 12) / max(1, len(active)))
            for subindex, item in enumerate(active):
                cx = center + (subindex - (len(active) - 1) / 2) * spacing
                amount = item[key]
                h = 175 * amount / maximum
                y = 327 - h
                svg.extend([
                    f'<rect x="{cx - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{h:.2f}" rx="3" fill="{item["color"]}"/>',
                    f'<text x="{cx:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="12" font-weight="650">{amount:.2f}</text>'
                ])
    legend = []
    seen = set()
    for group in groups:
        for value in group["values"]:
            if value["name"] not in seen:
                seen.add(value["name"])
                legend.append(value)
    start = 630 - sum(len(value["name"]) * 8 + 36 for value in legend) // 2
    for value in legend:
        svg.extend([
            f'<rect x="{start}" y="401" width="12" height="12" rx="2" fill="{value["color"]}"/>',
            f'<text x="{start + 18}" y="412" font-size="12">{escape(value["name"])}</text>',
        ])
        start += len(value["name"]) * 8 + 36
    svg.extend([
        f'<text x="24" y="446" font-size="13">{escape(foot1)}</text>',
        f'<text x="24" y="465" font-size="13">{escape(foot2)}</text>',
        '</g></svg>',
    ])
    (DOCS / path).write_text("\n".join(svg) + "\n")


def main(throughput):
    six = stage6(throughput)
    def item(name, color, lat=None, thr=None):
        return {"name": name, "color": color, "lat": lat, "thr": thr}

    precision = [
        item("PyTorch", GRAY, 51.93, 20.38),
        item("FP32 off", BLUE, 23.63, 53.36),
        item("FP32 on", TEAL, 17.74, 66.43),
        item("FP16 mix", ORANGE, 16.45, 74.35),
        item("Stage 6 FFN", FFN, six["ffn_p95"], six["ffn_throughput"]),
    ]
    render(
        "precision-performance.svg", "PyTorch vs. TensorRT · Stage 6 included",
        "RTX 4090 · 1024×1024 · batch 1 · historical 500-JPEG runs",
        [{"name": row["name"], "values": [row]} for row in precision],
        "Stage 2–4 P95 starts at JPEG processing; Stage 6 starts at slot dispatch. Stage 5 uses peak S3/S3/S2 throughput.",
        "All throughput uses the full offline run. Stage 6 FFN uses B1/S2, with PQ 53.1856 and 26/32 image agreement.",
    )

    stages = [
        {"name": "Stage 2", "values": [item("PyTorch", GRAY, 51.93, 20.38)]},
        {"name": "Stage 3", "values": [
            item("FP32 off", BLUE, 35.18, 31.18),
            item("FP32 on", TEAL, 29.92, 37.06),
            item("FP16 mix", ORANGE, 28.71, 38.78),
        ]},
        {"name": "Stage 4", "throughput_spacing": 26, "values": [
            item("FP32 off", BLUE, 23.63, 43.65),
            item("FP32 on", TEAL, 17.74, 58.26),
            item("FP16 mix", ORANGE, 16.45, 63.04),
        ]},
        {"name": "Stage 5", "values": [
            item("FP32 off", BLUE, thr=53.36),
            item("FP32 on", TEAL, thr=66.43),
            item("FP16 mix", ORANGE, thr=74.35),
        ]},
        {"name": "Stage 6", "values": [
            item("FPN FP16", FPN, six["fpn_p95"], six["fpn_throughput"]),
            item("FFN FP16", FFN, six["ffn_p95"], six["ffn_throughput"]),
        ]},
    ]
    render(
        "roadmap-performance.svg", "Inference performance by stage · Stage 2–6",
        "Stage 2 PyTorch · Stage 3 Python/TRT · Stage 4 native · Stage 5 streams · Stage 6 layer optimization",
        stages,
        "Stage 6 FPN and FFN use B1/S1 for P95 and B1/S2 for throughput. All throughput values use the full offline run.",
        "Stage 5 shows peak screening results. Use within-stage paired A/B to assess improvements.",
    )


if __name__ == "__main__":
    main(stage6_throughput())
