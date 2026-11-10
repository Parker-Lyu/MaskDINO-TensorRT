"""Compare the shipped, rebuilt control, and isolated FP16 FPN B1 engines."""

import hashlib
import json
from pathlib import Path
import subprocess

from stage5.measure import summarize
from stage5.validate import alignment


OUTPUT = Path("/output")
PREVIOUS = Path("/previous")
ENGINES = {name: Path("/plans") / name / "network.engine"
           for name in ("shipped", "control", "fpn-half")}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def native(engine, output, slots, mode, manifest):
    output.mkdir()
    command = [OUTPUT / "maskdino-slots", engine, "-", PREVIOUS / "perf-native-r1/things.txt",
               manifest, output, 1, slots, mode, 1, 0, 64]
    print("Run", *command, flush=True)
    subprocess.run([str(arg) for arg in command], check=True)


def verify_engines(engines=ENGINES):
    for name, engine in engines.items():
        build = json.loads(engine.with_name("build.json").read_text())
        with engine.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != build["engine_sha256"]:
            raise ValueError(f"{name} does not match build.json")
        print(name, actual, build["onnx_sha256"])


def check_outputs():
    manifest = PREVIOUS / "native-align-fixed/manifest.tsv"
    shipped = OUTPUT / "shipped-align-s1"
    native(ENGINES["shipped"], shipped, 1, "predict", manifest)
    passed = True
    for name, slots in (("control", 1), ("fpn-half", 1), ("fpn-half", 2)):
        output = OUTPUT / f"{name}-align-s{slots}"
        native(ENGINES[name], output, slots, "predict", manifest)
        result = alignment(output, shipped)
        write_json(output / "validation.json", result)
        print(output.name, sum(row["passed"] for row in result["per_image"]), "/", result["images"])
        if name == "fpn-half":
            passed &= result["passed"]
    if not passed:
        raise SystemExit("FPN candidate failed the per-image alignment gate")


def benchmark():
    manifest = PREVIOUS / "perf-native-r1/manifest.tsv"
    variants = list(ENGINES)
    rows = []
    for round_index in range(len(variants)):
        order = variants[round_index:] + variants[:round_index]
        for slots in (1, 2) if round_index % 2 == 0 else (2, 1):
            for name in order:
                output = OUTPUT / f"{name}-s{slots}-r{round_index + 1}"
                native(ENGINES[name], output, slots, "benchmark", manifest)
                row = summarize(output)
                if row["completed"] != 500 or row["rejected"]:
                    raise RuntimeError(f"Incomplete timing run: {output}")
                rows.append({"engine": name, "slots": slots, "round": round_index + 1, **row})
                write_json(OUTPUT / "summary.json", rows)


if __name__ == "__main__":
    verify_engines()
    check_outputs()
    benchmark()
