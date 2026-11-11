"""Convert the six encoder FFN branches of the FPN-optimized B1 ONNX to FP16."""
import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def convert(model):
    graph = model.graph
    nodes = {node.name: node for node in graph.node}
    consumers = Counter(name for node in graph.node for name in node.input)
    consumers.update(value.name for value in graph.output)
    value_info = {value.name: value for value in graph.value_info}
    insert_before = {}
    insert_after = {}

    for index in range(6):
        prefix = f"/sem_seg_head/transformer/encoder/layers.{index}"
        names = ("linear1/MatMul", "linear1/Add", "Relu", "linear2/MatMul", "linear2/Add")
        chain = [nodes[f"{prefix}/{name}"] for name in names]
        assert [node.op_type for node in chain] == ["MatMul", "Add", "Relu", "MatMul", "Add"]
        for first, second in zip(chain, chain[1:]):
            output, = first.output
            assert output in second.input and consumers[output] == 1
            if output in value_info:
                value_info[output].type.tensor_type.elem_type = TensorProto.FLOAT16

        for node, previous in zip(chain, [None, *chain[:-1]]):
            casts = []
            for offset, name in enumerate(node.input):
                if previous is not None and name in previous.output:
                    continue
                cast_output = f"{node.name}/fp16_input_{offset}"
                casts.append(helper.make_node("Cast", [name], [cast_output], name=cast_output,
                                              to=TensorProto.FLOAT16))
                node.input[offset] = cast_output
            insert_before[node.name] = casts

        final = chain[-1]
        output, = final.output
        final.output[0] = f"{final.name}/fp16_output"
        insert_after[final.name] = helper.make_node("Cast", list(final.output), [output],
                                                   name=f"{final.name}/fp32_output", to=TensorProto.FLOAT)

    rewritten = []
    for node in graph.node:
        rewritten.extend(insert_before.get(node.name, ()))
        rewritten.append(node)
        if node.name in insert_after:
            rewritten.append(insert_after[node.name])
    del graph.node[:]
    graph.node.extend(rewritten)
    onnx.checker.check_model(model)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error(f"Output already exists: {args.destination}")
    model = onnx.load(args.source)
    convert(model)
    onnx.save(model, args.destination)


if __name__ == "__main__":
    main()
