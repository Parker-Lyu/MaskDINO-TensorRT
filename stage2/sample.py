"""Build the fixed COCO val sample from ground truth and JPEG metadata."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads((args.dataset / "annotations/panoptic_val2017.json").read_text())
    images = sorted(data["images"], key=lambda image: image["id"])
    segments = {ann["image_id"]: ann["segments_info"] for ann in data["annotations"]}
    things = {cat["id"] for cat in data["categories"] if cat["isthing"]}
    assert len(images) == len(segments) == 5000

    def measurements(image):
        annotations = segments[image["id"]]
        objects = [seg for seg in annotations if seg["category_id"] in things and not seg["iscrowd"]]
        return {
            "pixels": image["height"] * image["width"],
            "aspect": image["width"] / image["height"],
            "jpeg_bytes": (args.dataset / "val2017" / image["file_name"]).stat().st_size,
            "segments": len(annotations),
            "things": len(objects),
            "small_things": sum(seg["area"] < 32**2 for seg in objects),
        }

    metrics = {image["id"]: measurements(image) for image in images}
    pixel_quantiles = statistics.quantiles([m["pixels"] for m in metrics.values()], n=3)
    strata = defaultdict(list)
    for image_id, m in metrics.items():
        key = (sum(m["pixels"] > cut for cut in pixel_quantiles),
               sum(m["aspect"] > cut for cut in (0.9, 1.1)))
        strata[key].append(image_id)

    keys = sorted(strata)
    quotas = {key: len(strata[key]) // 10 for key in keys}
    by_remainder = sorted(keys, key=lambda key: (-(len(strata[key]) % 10), key))
    for key in by_remainder[:500 - sum(quotas.values())]:
        quotas[key] += 1
    seed = 20261106
    rng = random.Random(seed)
    selected = [image_id for key in keys for image_id in rng.sample(strata[key], quotas[key])]
    rng.shuffle(selected)
    assert len(set(selected)) == 500

    # Stress the smoke test with the largest originals, extreme shapes and crowds.
    smoke = []
    for metric in ("pixels", "aspect", "things", "segments"):
        ranked = sorted(metrics, key=lambda image_id: metrics[image_id][metric])
        for image_id in (ranked[:4] + ranked[-4:]):
            if image_id not in smoke:
                smoke.append(image_id)
    smoke.extend(rng.sample(sorted(set(metrics) - set(smoke)), 32 - len(smoke)))

    def summary(ids):
        return {
            name: {
                "mean": round(statistics.mean(metrics[i][name] for i in ids), 4),
                "median": statistics.median(metrics[i][name] for i in ids),
                "p95": statistics.quantiles([metrics[i][name] for i in ids], n=20)[18],
            }
            for name in next(iter(metrics.values()))
        }

    def small_share(ids):
        return sum(metrics[i]["small_things"] for i in ids) / sum(metrics[i]["things"] for i in ids)

    result = {
        "seed": seed,
        "pixel_cutoffs": pixel_quantiles,
        "aspect_cutoffs": [0.9, 1.1],
        "strata": {str(key): [len(strata[key]), quotas[key]] for key in keys},
        "full": summary(metrics),
        "sample": summary(selected),
        "small_thing_share": {"full": small_share(metrics), "sample": small_share(selected)},
        "image_ids": selected,
        "smoke_image_ids": smoke,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
