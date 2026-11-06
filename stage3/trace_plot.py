"""Render a matched request from two Nsight Systems SQLite exports."""
import argparse
from pathlib import Path
import sqlite3

from trace_summary import ACTIVITIES, STAGES, union

COLORS = {
    "decode": "#f2ae55", "preprocess": "#e27757", "H2D": "#935dc1",
    "preprocess_gpu": "#b48ad6", "network": "#3d83c3",
    "postprocess": "#2f9f80", "D2H": "#a47732",
}


def capture(path, request):
    with sqlite3.connect(path) as db:
        ranges = db.execute("""
            SELECT coalesce(e.text, s.value), e.start, e.end
            FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
            WHERE e.end IS NOT NULL ORDER BY e.start
        """).fetchall()
        if request is None:
            request = next(label for label, _, _ in ranges if label and label.startswith("image "))
        start, end = next((left, right) for label, left, right in ranges if label == request)
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        intervals = [
            (max(start, left), min(end, right))
            for kind in ACTIVITIES if (table := f"CUPTI_ACTIVITY_KIND_{kind}") in tables
            for left, right in db.execute(f"SELECT start, end FROM {table} WHERE start < ? AND end > ?", (end, start))
        ]
    host = [(label, left - start, right - start) for label, left, right in ranges
            if label in STAGES and start <= left < right <= end]
    assert host and intervals
    return request, end - start, host, [(left - start, right - start) for left, right in union(intervals)]


def plot(pytorch, tensorrt, output):
    request, *pt = capture(pytorch, None)
    _, *trt = capture(tensorrt, request)
    tick_ns = 5_000_000
    limit = ((max(pt[0], trt[0]) + tick_ns - 1) // tick_ns) * tick_ns
    origin, width = 155, 875

    def x(value):
        return origin + width * value / limit

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="345" viewBox="0 0 1080 345">',
           '<rect width="1080" height="345" rx="14" fill="#f5f8fb"/>',
           '<g font-family="system-ui,sans-serif" fill="#182d43">']

    def text(x_pos, y, label, size=13):
        svg.append(f'<text x="{x_pos}" y="{y}" font-size="{size}">{label}</text>')

    def band(left, right, y, color):
        svg.append(f'<rect x="{x(left):.3f}" y="{y}" width="{x(right)-x(left):.3f}" height="17" fill="{color}"/>')

    text(24, 34, "Nsight Systems · same COCO request · 1024×1024 / B1 / single slot", 23)
    text(24, 57, f"COCO {request} · first request after warm-up · each timeline starts at its own request boundary")
    for tick in range(0, limit + 1, tick_ns):
        svg.append(f'<path d="M{x(tick):.3f} 100V272" stroke="#dce4ec"/>')
        text(x(tick) - 5, 292, f"{tick / 1e6:g}")
    text(1039, 292, "ms")

    for index, (title, (duration, host, gpu)) in enumerate((("PyTorch FP32", pt), ("TensorRT FP32", trt))):
        y = 82 + index * 100
        text(24, y + 12, f"{title} · {duration / 1e6:.2f} ms", 16)
        text(24, y + 38, "Host NVTX")
        text(24, y + 63, "GPU activity")
        band(0, duration, y + 24, "#e7edf3")
        band(0, duration, y + 49, "#e7edf3")
        for name, left, right in host:
            band(left, right, y + 24, COLORS[name])
        for left, right in gpu:
            band(left, right, y + 49, "#213d59")

    for index, name in enumerate(COLORS):
        left = 24 + 145 * index
        svg.append(f'<rect x="{left}" y="313" width="12" height="12" rx="2" fill="{COLORS[name]}"/>')
        text(left + 19, 324, name)
    svg.append("</g></svg>")
    output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytorch", type=Path)
    parser.add_argument("tensorrt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.pytorch, args.tensorrt, args.output)
