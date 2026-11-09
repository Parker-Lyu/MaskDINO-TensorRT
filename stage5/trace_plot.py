"""Draw GPU stream activity from the archived B1/S1 and B1/S3 Nsight exports."""
import argparse
from collections import defaultdict
from pathlib import Path
import sqlite3

from stage5.trace_summary import activity_spans

WINDOW_MS = 60
COLORS = {"network": "#2875ae", "other": "#9baaba", "overlap": "#e28a32"}


def capture(path):
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        return db.execute("""
            WITH inference AS (
                SELECT n.start, n.end, n.globalTid,
                       row_number() OVER (ORDER BY n.start) AS request
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
                WHERE coalesce(n.text, s.value) = 'TensorRT inference' AND n.end > n.start
            ),
            launches AS (
                SELECT a.correlationId, i.request
                FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN inference i
                  ON a.globalTid = i.globalTid AND a.start >= i.start AND a.start < i.end
            )
            SELECT k.start, k.end, k.streamId, l.request
            FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN launches l
              ON k.correlationId = l.correlationId
            WHERE k.end > k.start ORDER BY k.start
        """).fetchall()


def plot(captures, output):
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="362" viewBox="0 0 1100 362">',
        '<rect width="1100" height="362" rx="14" fill="#f4f7fb"/>',
        '<g font-family="system-ui,sans-serif" fill="#172b43">',
    ]

    def label(x, y, value, size=13, weight=400):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{value}</text>')

    def x(timestamp, origin):
        return 168 + (timestamp - origin) * 885 / (WINDOW_MS * 1_000_000)

    def activity(y, intervals, origin, color):
        stop = origin + WINDOW_MS * 1_000_000
        segments = [f"M{x(max(left, origin), origin):.3f} {y}H{x(min(right, stop), origin):.3f}"
                    for left, right in intervals if left < stop and right > origin]
        if segments:
            svg.append(f'<path d="{" ".join(segments)}" stroke="{color}" stroke-width="9"/>')

    label(24, 33, "Nsight Systems · Cross-request TensorRT execution", 23, 700)
    label(24, 57, "RTX 4090 · 1024 × 1024 · FP32 · same B1 engine · first 60 ms of GPU activity")
    for tick in range(0, WINDOW_MS + 1, 10):
        pos = x(tick * 1_000_000, 0)
        svg.append(f'<path d="M{pos:g} 83V292" stroke="#d7e0ea"/>')
        label(pos - 5, 311, tick)
    label(24, 311, "Time (ms)")

    for title, kernels, top in zip(("B1/S1", "B1/S3"), captures, (83, 154)):
        origin = kernels[0][0]
        streams = sorted({stream for _, _, stream, _ in kernels})
        network = [(left, right, stream) for left, right, stream, request in kernels if request is not None]
        spans = list(activity_spans(network))
        busy = sum(right - left for left, right, count in spans if count)
        overlap = sum(right - left for left, right, count in spans if count >= 2)
        print(f"{title}: {len(network)} TensorRT kernels, {overlap / busy:.2%} overlap of network kernel activity")
        label(24, top + 11, title, 17, 700)
        for index, stream in enumerate(streams):
            y = top + 42 + index * 31
            label(24, y + 4, f"CUDA stream {stream}")
            activity(y, [(a, b) for a, b, s, req in kernels if s == stream and req is None],
                     origin, COLORS["other"])
            activity(y, [(a, b) for a, b, s in network if s == stream], origin, COLORS["network"])
            first = {}
            for left, _, s, request in kernels:
                if s == stream and request is not None:
                    first.setdefault(request, left)
            for request, start in first.items():
                if start < origin + (WINDOW_MS - 7) * 1_000_000:
                    label(x(start, origin), y - 8, f"Req {request}", 11, 600)
        if len(streams) > 1:
            label(24, 285, "Network overlap")
            activity(281, [(left, right) for left, right, count in spans if count >= 2],
                     origin, COLORS["overlap"])

    for pos, key, title in ((24, "network", "TensorRT kernel"), (230, "other", "Other CUDA kernel"),
                            (444, "overlap", "TensorRT kernels active on ≥2 streams")):
        svg.append(f'<rect x="{pos}" y="332" width="16" height="9" fill="{COLORS[key]}"/>')
        label(pos + 23, 341, title)
    svg.append("</g></svg>")
    output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b1s1", type=Path)
    parser.add_argument("b1s3", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot([capture(path) for path in (args.b1s1, args.b1s3)], args.output)
