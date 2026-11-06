"""Validate Stage 5 maps against the native B1 reference and score COCO PQ."""
import argparse
import json
from pathlib import Path

from stage3.alignment import LIMITS, panoptic_metrics
from stage4.protocol import evaluate, load_requests, native_prediction


def alignment(output, reference):
    _, images = load_requests("align")
    rows = []
    limits = LIMITS["fp32"]
    for image in images:
        baseline = native_prediction(reference, image)
        actual = native_prediction(output, image)
        metrics, _ = panoptic_metrics(baseline, actual)
        passed = (metrics["category_agreement"] >= limits["category_agreement"]
                  and not any(metrics["unmatched_segments"])
                  and min((match["iou"] for match in metrics["same_category_matches"]), default=1)
                  >= limits["segment_iou"])
        rows.append({"id": image["id"], "passed": passed, "metrics": metrics,
                     "identical_metadata": baseline["panoptic_seg"][1] == actual["panoptic_seg"][1]})
    return {"passed": all(row["passed"] for row in rows), "images": len(rows), "per_image": rows}


def full_pq(output):
    annotation, images = load_requests("evaluate")
    result = evaluate(output, annotation, images)
    stage4 = Path(__file__).resolve().parents[1] / "stage4/results/native-pq/result.json"
    baseline = json.loads(stage4.read_text())["scores"]
    deltas = {key: result["scores"][key] - baseline[key] for key in ("PQ", "PQ_th", "PQ_st")}
    result["delta_vs_stage4"] = deltas
    result["passed"] = all(delta >= -LIMITS["fp32"]["pq_delta"] for delta in deltas.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    result = alignment(args.output, args.reference) if args.reference else full_pq(args.output)
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print({key: value for key, value in result.items() if key != "per_image"}, flush=True)
    if not result["passed"]:
        raise SystemExit("Stage 5 correctness gate failed")


if __name__ == "__main__":
    main()
