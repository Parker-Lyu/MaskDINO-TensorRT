"""Measure the six-layer FP16 FFN candidate against the accepted FPN engine."""

import argparse
from pathlib import Path
import subprocess
import sys

from stage6.fpn_ab import OUTPUT, PREVIOUS, native, verify_engines, write_json
from stage5.measure import summarize
from stage5.validate import alignment, full_pq


ENGINES = {name: Path("/plans") / name / "network.engine" for name in ("fpn", "ffn-half")}


def check_outputs():
    manifest = PREVIOUS / "native-align-fixed/manifest.tsv"
    reference = OUTPUT / "fpn-align-s1"
    native(ENGINES["fpn"], reference, 1, "predict", manifest)
    for slots in (1, 2):
        output = OUTPUT / f"ffn-half-align-s{slots}"
        native(ENGINES["ffn-half"], output, slots, "predict", manifest)
        report = alignment(output, reference)
        write_json(output / "validation.json", report)
        print(output.name, sum(row["passed"] for row in report["per_image"]), "/", report["images"])


def benchmark():
    manifest = PREVIOUS / "perf-native-r1/manifest.tsv"
    rows = []
    for round_number in (1, 2, 3):
        order = tuple(ENGINES) if round_number % 2 else tuple(reversed(ENGINES))
        for slots in (1, 2) if round_number % 2 else (2, 1):
            for name in order:
                output = OUTPUT / f"{name}-s{slots}-r{round_number}"
                native(ENGINES[name], output, slots, "benchmark", manifest)
                report = summarize(output)
                if report["completed"] != 500 or report["rejected"]:
                    raise RuntimeError(f"Incomplete timing run: {output}")
                rows.append({"engine": name, "slots": slots, "round": round_number, **report})
                write_json(OUTPUT / "summary.json", rows)


def evaluate():
    manifest_dir = OUTPUT / "coco"
    manifest_dir.mkdir()
    engine = ENGINES["ffn-half"]
    subprocess.run([sys.executable, "-m", "stage4.protocol", "manifest", "evaluate",
                    manifest_dir, engine], check=True)
    for slots in (1, 2):
        output = OUTPUT / f"ffn-half-pq-s{slots}"
        native(engine, output, slots, "predict", manifest_dir / "manifest.tsv")
        report = full_pq(output, reference="pytorch")
        write_json(output / "validation.json", report)
        print(output.name, report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("screen", "pq"))
    args = parser.parse_args()
    verify_engines(ENGINES)
    if args.phase == "pq":
        evaluate()
    else:
        check_outputs()
        benchmark()


if __name__ == "__main__":
    main()
