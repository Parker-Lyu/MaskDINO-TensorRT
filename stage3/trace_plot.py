"""Render CPU ranges, synchronization calls and attributed GPU work for a matched request."""
import argparse
from collections import defaultdict
from html import escape
from pathlib import Path
import sqlite3

from trace_summary import ACTIVITIES, STAGES, STAGE_BY_LABEL, union

COLORS = {
    "decode": "#f2ae55", "preprocess": "#e27757", "H2D": "#935dc1",
    "preprocess_gpu": "#b48ad6", "network": "#3d83c3",
    "postprocess": "#2f9f80", "D2H": "#a47732", "unattributed": "#8894a0",
}
SYNC = "#b83d50"
WAIT = "#a6aab8"


def capture(path, request, split_legacy_output=False):
    with sqlite3.connect(path) as db:
        ranges = db.execute("""
            SELECT coalesce(e.text, s.value), e.start, e.end
            FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
            WHERE e.end IS NOT NULL ORDER BY e.start
        """).fetchall()
        if request is None:
            request = next(label for label, _, _ in ranges if label and label.startswith("image "))
        start, end = next((left, right) for label, left, right in ranges if label == request)
        stages = [(STAGE_BY_LABEL.get(label, label), left, right) for label, left, right in ranges]
        host = [(label, left, right) for label, left, right in stages
                if label in STAGES and start <= left < right <= end]
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        api = [
            row
            for kind in ("RUNTIME", "DRIVER")
            if (table := f"CUPTI_ACTIVITY_KIND_{kind}") in tables
            for row in db.execute(f"""
                SELECT s.value, a.start, a.end, a.correlationId FROM {table} a
                JOIN StringIds s ON a.nameId = s.id WHERE a.start >= ? AND a.start < ?
            """, (start, end))
        ]
        correlation = {
            event_id: next((label for label, left, right in host if left <= submitted < right), "unattributed")
            for _, submitted, _, event_id in api
        }
        gpu = defaultdict(list)
        for kind in ACTIVITIES:
            table = f"CUPTI_ACTIVITY_KIND_{kind}"
            if table in tables:
                for left, right, event_id in db.execute(
                    f"SELECT start, end, correlationId FROM {table} WHERE start < ? AND end > ?", (end, start)
                ):
                    gpu[correlation.get(event_id, "unattributed")].append((max(start, left), min(end, right)))

    assert host and gpu
    if split_legacy_output:
        output = [(left, right) for name, left, right in host if name == "D2H"]
        assert len(output) == 1 and not any(name == "output_wait" for name, _, _ in host)
        left, right = output[0]
        output_api = [(name, a, b) for name, a, b, _ in api if left <= a < b <= right]
        copies = union((a, b) for name, a, b in output_api if name.startswith("cudaMemcpyAsync"))
        waits = union((a, b) for name, a, b in output_api if name.startswith("cudaStreamSynchronize"))
        assert copies and len(waits) == 1
        host = [(name, a, b) for name, a, b in host if name != "D2H"]
        host.extend(("D2H", a, b) for a, b in copies)
        host.extend(("output_wait", a, b) for a, b in waits)
    sync = union((left, min(end, right)) for name, left, right, _ in api if "Synchronize" in name)
    return request, start, end, host, sync, {name: union(intervals) for name, intervals in gpu.items()}


def duration_ms(intervals):
    return sum(right - left for left, right in intervals) / 1e6


def plot(pytorch, tensorrt, output, labels=("PyTorch FP32", "TensorRT FP32"), split_legacy_output=False):
    request, *pt = capture(pytorch, None)
    _, *trt = capture(tensorrt, request, split_legacy_output)
    split_output = any(name == "output_wait" for trace in (pt, trt) for name, _, _ in trace[2])
    colors = {**COLORS, "output_wait": WAIT}
    tick_ns = 5_000_000
    span = max(end - start for start, end, *_ in (pt, trt))
    limit = ((span + tick_ns - 1) // tick_ns) * tick_ns
    origin, width = 177, 900

    def x(offset):
        return origin + width * offset / limit

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="449" viewBox="0 0 1120 449">',
           '<rect width="1120" height="449" rx="14" fill="#f5f8fb"/>',
           '<g font-family="system-ui,sans-serif" fill="#182d43">']

    def text(x_pos, y, label, size=13):
        svg.append(f'<text x="{x_pos}" y="{y}" font-size="{size}">{escape(label)}</text>')

    def band(left, right, y, color):
        svg.append(f'<rect x="{x(left):.3f}" y="{y}" width="{x(right)-x(left):.3f}" height="15" fill="{color}"/>')

    text(24, 34, "Nsight Systems · CPU waiting vs. GPU execution", 23)
    text(24, 57, f"COCO {request} · first request after warm-up · 1024×1024 / B1 / one slot")
    for tick in range(0, limit + 1, tick_ns):
        svg.append(f'<path d="M{x(tick):.3f} 100V341" stroke="#dce4ec"/>')
        text(x(tick) - 5, 363, f"{tick / 1e6:g}")
    text(1087, 363, "ms")

    for index, (title, (start, end, host, sync, gpu)) in enumerate(
        zip(labels, (pt, trt))
    ):
        y = 78 + index * 134
        text(24, y + 12, f"{title} · {(end - start) / 1e6:.2f} ms request", 16)
        for offset, label in ((24, "CPU ranges" if split_legacy_output else "CPU NVTX"),
                              (46, "CPU sync API"), (68, "GPU by submit stage")):
            text(24, y + offset + 12, label)
            band(0, end - start, y + offset, "#e7edf3")
        for name, left, right in host:
            band(left - start, right - start, y + 24, colors[name])
        for left, right in sync:
            band(left - start, right - start, y + 46, SYNC)
        for name, intervals in gpu.items():
            for left, right in intervals:
                band(left - start, right - start, y + 68, colors[name])
        wait = next(((left, right) for name, left, right in host if name == "output_wait"), None)
        stage = "postprocess"
        if wait is None:
            left, right = next((left, right) for name, left, right in host if name == stage)
            note = f"CPU postprocess {(right - left) / 1e6:.2f} ms"
            gpu_note = f"postprocess GPU activity {duration_ms(gpu[stage]):.2f} ms"
        else:
            left, right = wait
            note = f"CPU output wait {(right - left) / 1e6:.2f} ms"
            gpu_note = f"GPU D2H copies {duration_ms(gpu['D2H']):.3f} ms"
        overlap = duration_ms((max(a, left), min(b, right))
                              for a, b in gpu["network"] if a < right and b > left)
        text(origin, y + 108, f"{note}  ·  overlaps {overlap:.2f} ms of network GPU activity  ·  {gpu_note}")

    legend = {**COLORS, "sync API": SYNC}
    if split_output:
        legend["output wait"] = WAIT
    for index, (name, color) in enumerate(legend.items()):
        left = 24 + (106 if split_output else 118) * index
        svg.append(f'<rect x="{left}" y="389" width="12" height="12" rx="2" fill="{color}"/>')
        text(left + 17, 400, name)
    footer = "GPU colors follow CUDA API correlation IDs. CPU sync time overlaps GPU execution."
    if split_legacy_output:
        footer = "Legacy C++ D2H range split using CUDA API events. GPU colors follow CUDA API correlation IDs."
    text(24, 428, footer)
    svg.append("</g></svg>")
    output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytorch", type=Path)
    parser.add_argument("tensorrt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--labels", nargs=2, default=("PyTorch FP32", "TensorRT FP32"))
    parser.add_argument("--split-legacy-output", action="store_true",
                        help="Split the second trace's legacy D2H NVTX range using its CUDA API events")
    args = parser.parse_args()
    plot(args.pytorch, args.tensorrt, args.output, args.labels, args.split_legacy_output)
