"""Check the engine on identical inputs, before and after shared postprocessing."""
from collections import Counter
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from stage2.run import postprocess_batch, prepare_batch, pytorch_network
from stage3.run import OUTPUT, SIZE, dump

LIMITS = {
    "fp32": {"category_agreement": 0.999, "segment_iou": 0.98, "pq_delta": 0.05},
    "mixed": {"category_agreement": 0.99, "segment_iou": 0.90, "pq_delta": 0.20},
}


def tensor_errors(reference, current):
    result = {}
    for name in ("logits", "masks", "boxes"):
        a, b = (out[f"pred_{name}"].float() for out in (reference, current))
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(), name
        delta = a - b
        result[name] = {
            "max_abs": delta.abs().max().item(),
            "mean_abs": delta.abs().mean().item(),
            "relative_l2": (delta.norm() / a.norm().clamp_min(1e-8)).item(),
        }
    return result


def query_match(reference, current):
    def features(output):
        masks = F.adaptive_avg_pool2d(output["pred_masks"].float(), (16, 16))[0].flatten(1)
        return F.normalize(masks), output["pred_logits"][0].float().sigmoid()

    masks_a, logits_a = features(reference)
    masks_b, logits_b = features(current)
    cost = 1 - masks_a @ masks_b.T + torch.cdist(logits_a, logits_b, p=2) / logits_a.shape[1]**0.5
    rows, columns = linear_sum_assignment(cost.cpu().numpy())
    assert np.array_equal(rows, np.arange(len(columns)))
    return columns


def panoptic_metrics(reference, current):
    (map_a, seg_a), (map_b, seg_b) = reference["panoptic_seg"], current["panoptic_seg"]
    a, b = map_a.numpy(), map_b.numpy()
    metadata_a = {seg["id"]: (seg["category_id"], seg["isthing"]) for seg in seg_a}
    metadata_b = {seg["id"]: (seg["category_id"], seg["isthing"]) for seg in seg_b}
    stride = max(metadata_b, default=0) + 1
    counts = np.bincount((a * stride + b).ravel(), minlength=(max(metadata_a, default=0) + 1) * stride)
    intersection = counts.reshape(-1, stride)
    area_a, area_b = intersection.sum(axis=1), intersection.sum(axis=0)
    pairs = []
    category_correct = int(intersection[0, 0])
    for i, category in metadata_a.items():
        for j, candidate in metadata_b.items():
            if category == candidate:
                overlap = int(intersection[i, j])
                category_correct += overlap
                union = int(area_a[i] + area_b[j]) - overlap
                if overlap:
                    pairs.append((overlap / union, i, j))
    match_a, match_b, matched = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in match_a and j not in match_b:
            matched.append({"reference_id": i, "engine_id": j, "iou": iou})
            match_a.add(i)
            match_b.add(j)
    category_a, category_b = np.full(a.shape, -1), np.full(b.shape, -1)
    for id_map, categories, metadata in ((a, category_a, metadata_a), (b, category_b, metadata_b)):
        for seg_id, (category, isthing) in metadata.items():
            categories[id_map == seg_id] = 2 * category + int(isthing)
    metrics = {
        "category_agreement": category_correct / a.size,
        "segment_counts": [len(seg_a), len(seg_b)],
        "metadata_equal": seg_a == seg_b,
        "category_metadata_counts_equal": Counter(metadata_a.values()) == Counter(metadata_b.values()),
        "unmatched_segments": [len(seg_a) - len(match_a), len(seg_b) - len(match_b)],
        "same_category_matches": matched,
    }
    return metrics, category_a != category_b


def validate(model, network, records, ids, read, backend, report):
    limits = LIMITS[backend]
    observations = []
    for image_id in ids:
        images, inputs = prepare_batch(model, [read(image_id)], SIZE)
        reference = pytorch_network(model, images)
        current = network(images)
        try:
            direct = tensor_errors(reference, current)
            query_relative = direct["masks"]["relative_l2"]
            matched = None
            if query_relative > 0.02:
                permutation = query_match(reference, current)
                matched = {
                    "moved": int((permutation != np.arange(len(permutation))).sum()),
                    "errors": tensor_errors(reference, {key: tensor[:, permutation] for key, tensor in current.items()}),
                }
            predictions = [
                postprocess_batch(model, [records[image_id]], inputs, output, SIZE, time.perf_counter(), False)[0][0]
                for output in (reference, current)
            ]
            panoptic, diff = panoptic_metrics(*predictions)
            ious = [match["iou"] for match in panoptic["same_category_matches"]]
            passed = (
                panoptic["category_agreement"] >= limits["category_agreement"]
                and not any(panoptic["unmatched_segments"])
                and min(ious, default=1) >= limits["segment_iou"]
            )
            observations.append({"image_id": image_id, "direct": direct, "matched": matched, "panoptic": panoptic, "passed": passed})
        except AssertionError as error:
            observations.append({"image_id": image_id, "passed": False, "tensor_check_error": str(error)})
            predictions, diff, passed = None, None, False
        if not passed:
            def cpu(outputs):
                return {name: outputs[f"pred_{name}"].cpu() for name in ("logits", "masks", "boxes")}
            torch.save(
                {"input": images.cpu(), "pytorch": cpu(reference), "engine": cpu(current), "panoptic": predictions},
                OUTPUT / f"anomaly-{image_id}.pt",
            )
            if diff is not None:
                Image.fromarray(diff.astype(np.uint8) * 255).save(OUTPUT / f"anomaly-{image_id}-category.png")
                id_diff = predictions[0]["panoptic_seg"][0] != predictions[1]["panoptic_seg"][0]
                Image.fromarray(id_diff.numpy().astype(np.uint8) * 255).save(OUTPUT / f"anomaly-{image_id}-id.png")
    report.update({"limits": limits, "per_image": observations, "passed": all(item["passed"] for item in observations)})
    dump("result.json", report)
    print([(item["image_id"], item.get("panoptic", {}).get("category_agreement")) for item in observations if not item["passed"]])
    if not report["passed"]:
        raise RuntimeError("32-image alignment gate failed")
