"""Separate logit and mask changes on a failed FPN alignment input."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from stage3.alignment import panoptic_metrics, tensor_errors
from stage3.trt_backend import Engine
from stage4.protocol import Geometry, NativePanoptic


def compare_tensors(reference, actual):
    result = tensor_errors(reference, actual)
    for key, values in result.items():
        values["exact"] = torch.equal(reference[f"pred_{key}"], actual[f"pred_{key}"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("geometry", type=int, nargs=4)
    parser.add_argument("things", type=Path)
    parser.add_argument("restore_library", type=Path)
    parser.add_argument("engines", type=Path, nargs=3, metavar="ENGINE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    things = [int(value) for value in args.things.read_text().split()]
    metadata = SimpleNamespace(thing_dataset_id_to_contiguous_id=dict(enumerate(things)))
    panoptic = NativePanoptic(args.restore_library, SimpleNamespace(metadata=metadata))
    image = torch.from_numpy(np.fromfile(args.input, dtype=np.float32).reshape(1, 3, 1024, 1024)).cuda()
    geometry = Geometry(*args.geometry)
    names = ("shipped", "control", "fpn-half")

    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        engines = {name: Engine(path) for name, path in zip(names, args.engines)}
        outputs = {name: engine(image) for name, engine in engines.items()}
        reference = outputs["shipped"]
        prediction = panoptic(reference["pred_logits"], reference["pred_masks"], geometry)
        tensors = {name: compare_tensors(reference, outputs[name]) for name in names[1:]}
        paired = compare_tensors(outputs["control"], outputs["fpn-half"])

        swaps = {}
        for logits_name in names:
            for masks_name in names:
                actual = panoptic(outputs[logits_name]["pred_logits"], outputs[masks_name]["pred_masks"], geometry)
                metrics, _ = panoptic_metrics(prediction, actual)
                metrics.pop("same_category_matches")
                swaps[f"{logits_name}-logits/{masks_name}-masks"] = metrics
    args.output.write_text(json.dumps({
        "tensors_vs_shipped": tensors, "candidate_tensors_vs_control": paired,
        "swaps_vs_shipped": swaps,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
