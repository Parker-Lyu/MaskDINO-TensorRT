"""Run Stage 5 scenarios in the reference container."""
import argparse
from hashlib import file_digest
import itertools
import json
from pathlib import Path
import statistics
import subprocess

from stage5.measure import arrivals, summarize

OUTPUT = Path("/output")
PREVIOUS = Path("/previous")
B1 = Path("/b1.engine")
BATCH_ENGINES = Path("/batch-engines")
BASELINE = PREVIOUS / "perf-native-r1"
ALIGN = PREVIOUS / "native-align-fixed"
BINARY = OUTPUT / "maskdino-slots"


def engine_path(batch):
    if batch == 1:
        return B1
    directory = BATCH_ENGINES if BATCH_ENGINES.is_dir() else Path("/project/stage3/runs")
    return directory / f"build-b{batch}/network.engine"


def verify_batch_engines(configs):
    if not BATCH_ENGINES.is_dir():
        return
    b1 = json.loads(Path("/b1-build.json").read_text())
    for batch in sorted({batch for batch, _ in configs} - {1}):
        engine = engine_path(batch)
        build = json.loads(engine.with_name("build.json").read_text())
        policy = ("precision", "tf32", "tensorrt")
        if any(build[key] != b1[key] for key in policy):
            raise ValueError(f"{engine} precision or TensorRT version differs from B1")
        image = b1["io"]["image"]["shape"]
        if build["io"]["image"]["shape"] != [batch, *image[1:]]:
            raise ValueError(f"{engine} has an unexpected input shape")
        with engine.open("rb") as stream:
            sha = file_digest(stream, "sha256").hexdigest()
        if sha != build["engine_sha256"]:
            raise ValueError(f"{engine} does not match build.json")
        print("Verified batch engine", engine, sha, *(build[key] for key in policy))


def native(output, batch, slots, mode, manifest, workload, delay=0, prefix=()):
    output.mkdir()
    engine = engine_path(batch)
    if not engine.is_file():
        raise FileNotFoundError(engine)
    command = [*prefix, BINARY, engine, B1, BASELINE / "things.txt", manifest, output,
               batch, slots, mode, workload, delay, 64]
    print("Run", *command, flush=True)
    subprocess.run([str(arg) for arg in command], check=True)


def configuration(text):
    batch, slots = map(int, text.lower().split("s"))
    if batch not in (1, 2, 4, 8) or not 1 <= slots <= 16:
        raise ValueError(text)
    return batch, slots


def name(config):
    return f"b{config[0]}s{config[1]}"


def benchmark(output, config, repeats):
    native(output, *config, "benchmark", BASELINE / "manifest.tsv", repeats)
    return summarize(output)


def screen(configs, repeats, *, include_stage4=True):
    rows = []
    for index in range(3):
        order = configs[index:] + configs[:index]
        for config in order:
            output = OUTPUT / f"{name(config)}-r{index + 1}"
            rows.append({"config": name(config), "round": index + 1, **benchmark(output, config, repeats)})
        if repeats == 1 and include_stage4:
            output = OUTPUT / f"stage4-r{index + 1}"
            output.mkdir()
            subprocess.run([BASELINE / "maskdino-native", B1, BASELINE / "things.txt",
                            BASELINE / "manifest.tsv", "benchmark", output], check=True)
    (OUTPUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for config in configs:
        values = [row["makespan_images_per_second"] for row in rows if row["config"] == name(config)]
        print(name(config), values, "median", statistics.median(values))


def check(configs):
    from stage5.validate import alignment
    for config in configs:
        output = OUTPUT / name(config)
        native(output, *config, "predict", ALIGN / "manifest.tsv", 1)
        result = alignment(output, ALIGN)
        (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name(config), sum(row["passed"] for row in result["per_image"]), "/", result["images"])
        if not result["passed"]:
            raise RuntimeError(f"{name(config)} correctness gate failed")
    if (4, 2) in configs:
        output = OUTPUT / "tail"
        manifest = OUTPUT / "tail-manifest.tsv"
        lines = (ALIGN / "manifest.tsv").read_text().splitlines()[:-1]
        manifest.write_text("\n".join(lines) + "\n")
        native(output, 4, 2, "predict", manifest, 1)
        result = alignment(output, ALIGN, {int(line.split()[0]) for line in lines})
        (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        report = summarize(output)
        assert result["passed"] and report["batch_dispatch_histogram"] == {4: 7, 1: 3}
        print("Offline tail:", report["batch_dispatch_histogram"])
    if (2, 2) in configs:
        output = OUTPUT / "mixed"
        trace = OUTPUT / "mixed-arrivals.txt"
        trace.write_text("".join(f"{(i // 3) * 160 + (i % 3 == 2) * 65 + 100}\n" for i in range(32)))
        native(output, 2, 2, "online-predict", ALIGN / "manifest.tsv", trace, 5)
        result = alignment(output, ALIGN)
        (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        report = summarize(output, 2.0)
        assert result["passed"] and report["batch_dispatch_histogram"].keys() == {1, 2}
        print("Mixed B1/B2:", report["batch_dispatch_histogram"])


def online(configs, delays, shapes, rates, seconds, seeds):
    rows = []
    for seed, shape, rate in itertools.product(seeds, shapes, rates):
        trace = OUTPUT / f"{shape}-{rate:g}-{seed}.arrivals"
        arrivals(trace, rate, seconds, shape, seed)
        for config in configs:
            for delay in (delays if config[0] > 1 else [0]):
                output = OUTPUT / f"{trace.stem}-{name(config)}-{delay:g}ms"
                native(output, *config, "online", BASELINE / "manifest.tsv", trace, delay)
                row = dict(config=name(config), delay_ms=delay, shape=shape, rate=rate, seed=seed,
                           **summarize(output, seconds))
                rows.append(row)
                with (OUTPUT / "summary.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(output.name, "P99", row["latency_ms"]["p99"], "rejected", row["rejected"],
                      "sustained", row["sustained"])


def pq(configs, reference):
    from stage5.validate import full_pq

    manifest_dir = OUTPUT / "coco"
    manifest_dir.mkdir()
    subprocess.run(["python", "-m", "stage4.protocol", "manifest", "evaluate", manifest_dir, B1], check=True)
    for config in configs:
        output = OUTPUT / name(config)
        native(output, *config, "predict", manifest_dir / "manifest.tsv", 1)
        result = full_pq(output, reference=reference)
        (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name(config), result)
        if not result["passed"]:
            raise RuntimeError(f"{name(config)} COCO gate failed")


def trace(configs):
    for config in configs:
        path = OUTPUT / name(config)
        prefix = ("nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                  "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--output", OUTPUT / name(config))
        native(path, *config, "trace", ALIGN / "manifest.tsv", 1, prefix=prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "screen", "offline", "online", "pq", "trace"))
    parser.add_argument("configs", nargs="+", type=configuration, help="Example: 2s2")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skip-stage4", action="store_true", help="Reuse existing Stage 4 serial benchmarks")
    parser.add_argument("--pq-reference", choices=("stage4", "pytorch"), default="stage4")
    parser.add_argument("--rates", type=float, nargs="+")
    parser.add_argument("--delays", type=float, nargs="+", default=[0])
    parser.add_argument("--shapes", choices=("uniform", "poisson", "burst"), nargs="+", default=["poisson"])
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[47])
    args = parser.parse_args()
    verify_batch_engines(args.configs)
    if args.phase == "check":
        check(args.configs)
    elif args.phase in ("screen", "offline"):
        screen(args.configs, args.repeats, include_stage4=not args.skip_stage4)
    elif args.phase == "online":
        if not args.rates or min(args.rates) <= 0 or args.seconds <= 0:
            parser.error("Online requires positive rates and duration")
        online(args.configs, args.delays, args.shapes, args.rates, args.seconds, args.seeds)
    elif args.phase == "pq":
        pq(args.configs, args.pq_reference)
    else:
        trace(args.configs)


if __name__ == "__main__":
    main()
