"""Build a single-convolution FP16 ONNX candidate from the mixed B1 export."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


CONV = "/sem_seg_head/layer_1/Conv"


def use_half_fpn_conv(model):
    graph = model.graph
    matches = [(i, node) for i, node in enumerate(graph.node) if node.name == CONV]
    if len(matches) != 1:
        raise ValueError(f"Expected one {CONV} node, found {len(matches)}")
    index, conv = matches[0]
    if conv.op_type != "Conv" or len(conv.input) != 2 or len(conv.output) != 1:
        raise ValueError(f"Expected a bias-free Conv at {CONV}")
    activation, weight_name = conv.input
    output = conv.output[0]
    consumers = {node.name for node in graph.node if output in node.input}
    if consumers != {"/sem_seg_head/layer_1/norm/Reshape", "/sem_seg_head/layer_1/norm/Shape"}:
        raise ValueError(f"Unexpected FPN Conv output consumers: {consumers}")

    weights = next((w for w in graph.initializer if w.name == weight_name), None)
    if weights is None or weights.data_type != TensorProto.FLOAT or list(weights.dims) != [256, 256, 3, 3]:
        raise ValueError("Expected FP32 256 × 256 × 3 × 3 Conv weights")
    if sum(name == weight_name for node in graph.node for name in node.input) != 1:
        raise ValueError("Expected weights exclusive to the FPN Conv")

    prefix = "/sem_seg_head/layer_1/isolated_fp16"
    half_activation = f"{prefix}/activation"
    half_output = f"{prefix}/output"
    half_weights = numpy_helper.to_array(weights).astype("float16")
    weights.CopyFrom(numpy_helper.from_array(half_weights, name=weight_name))
    conv.input[0] = half_activation
    conv.output[0] = half_output
    graph.node.insert(index, helper.make_node("Cast", [activation], [half_activation],
                                               name=f"{prefix}/to_half", to=TensorProto.FLOAT16))
    graph.node.insert(index + 2, helper.make_node("Cast", [half_output], [output],
                                                   name=f"{prefix}/to_float", to=TensorProto.FLOAT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    model = onnx.load(args.source)
    use_half_fpn_conv(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
