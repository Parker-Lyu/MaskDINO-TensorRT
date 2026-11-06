"""Aggregate the paired single-slot benchmarks and draw the cumulative comparison."""
import argparse
from html import escape
import json
from pathlib import Path
from statistics import median

METRICS = ("network_ms_p50", "network_ms_p95", "e2e_ms_p50", "e2e_ms_p95",
           "images_per_second", "sampled_peak_process_used_bytes")
NAMES = {"pytorch": "PyTorch FP32", "reference": "TensorRT + Python", "native": "TensorRT + C++"}


def read(path):
    return json.loads(path.read_text())


def performance(results):
    summary = {}
    order = None
    engine = None
    for backend in NAMES:
        rounds = [read(results / f"perf-{backend}-r{i}" / "result.json") for i in range(1, 4)]
        for run in rounds:
            assert (run["images"], run["failures"], run["warmup"], run["size"], run["batch"], run["slots"]) == (
                500, 0, 20, 1024, 1, 1)
            assert run["memory_scope"] == "NVML CUDA compute process (host PID)"
            assert run["memory_sample_interval_ms"] == 20
            ids = [sample["image_id"] for sample in run["samples"]]
            order = order or ids
            assert ids == order
            if backend != "pytorch":
                engine = engine or run["engine_sha256"]
                assert run["engine_sha256"] == engine
        summary[backend] = {
            key: dict(median=median(values), minimum=min(values), maximum=max(values), rounds=values)
            for key in METRICS if (values := [run[key] for run in rounds])
        }
    return summary


def plot(results, perf, scores):
    root = results.resolve().parents[1]
    official = 52.9567
    stage2 = read(root / "stage2/results/evaluate/1024-4/result.json")["scores"]["PQ"]
    pt = read(root / "stage3/results/pt-pq/result.json")["scores"]["PQ"]
    trt = read(root / "stage3/results/fp32-pq-rerun/result.json")["scores"]["PQ"]
    panels = [
        ("COCO val2017 PQ", "", [("Stage 1 official", official), ("Stage 2 B4", stage2),
                                (NAMES["pytorch"], pt), (NAMES["reference"], trt), (NAMES["native"], scores["PQ"])]),
        *[(title, unit, [(name, perf[key][metric]["median"]) for key, name in NAMES.items()])
          for title, unit, metric in (
              ("End-to-end P95", " ms", "e2e_ms_p95"),
              ("Offline throughput", " img/s", "images_per_second"),
          )],
        ("Peak process VRAM", " GiB", [(name, perf[key]["sampled_peak_process_used_bytes"]["median"] / 2**30)
                                     for key, name in NAMES.items()]),
    ]
    colors = dict(zip(NAMES.values(), ("#457caa", "#168477", "#a85b36")))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="564" viewBox="0 0 1060 564">',
           '<rect width="1060" height="564" rx="16" fill="#f3f6fa"/>',
           '<g font-family="system-ui, sans-serif" fill="#172b43">']

    def label(x, y, content, size, weight=400):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{escape(content)}</text>')

    label(30, 38, "MaskDINO R50 panoptic · RTX 4090", 24, 700)
    label(30, 62, "Performance · 1024×1024 · batch 1 · single slot · median of 3 rounds", 13)
    for index, (title, unit, rows) in enumerate(panels):
        x, y = 24 + 512 * (index % 2), 82 + 236 * (index // 2)
        svg.append(f'<rect x="{x}" y="{y}" width="500" height="218" rx="10" fill="white"/>')
        label(x + 18, y + 30, title, 17, 650)
        ceiling = max(value for _, value in rows)
        for i, (name, value) in enumerate(rows):
            cy = y + 58 + i * 30
            label(x + 18, cy + 4, name, 12)
            color = colors.get(name, "#788a9e")
            svg.append(f'<rect x="{x + 155}" y="{cy - 8}" width="{230 * value / ceiling:.2f}" height="15" rx="3" fill="{color}"/>')
            label(x + 398, cy + 4, f"{value:.2f}{unit}" if unit else f"{value:.4f}", 12, 550)
    svg.append("</g></svg>")
    (results / "comparison.svg").write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    perf = performance(args.results)
    accuracy = read(args.results / "native-pq/result.json")
    assert accuracy["passed"] and accuracy["unique_prediction_ids"] == 5000
    (args.results / "summary.json").write_text(json.dumps(dict(performance=perf, accuracy=accuracy), indent=2) + "\n")
    plot(args.results, perf, accuracy["scores"])
