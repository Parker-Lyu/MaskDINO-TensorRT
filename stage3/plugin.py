"""Lower the upstream MSDA autograd op to NVIDIA TensorRT MSDA v2."""
import torch
from torch.onnx import symbolic_helper

from maskdino.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDeformAttnFunction


def install(batch=1):
    def symbolic(graph, value, shapes, starts, locations, weights, _step):
        result = graph.op(
            "trt::MultiscaleDeformableAttnPlugin_TRT",
            value, graph.op("Cast", shapes, to_i=6), graph.op("Cast", starts, to_i=6),
            locations, weights, plugin_version_s="2", plugin_namespace_s="",
        )
        # Legacy autocast export can omit static shape metadata on Cast outputs.
        heads, channels = 8, 32
        value_sizes = symbolic_helper._get_tensor_sizes(value)
        if value_sizes is not None:
            for index, expected in ((0, batch), (2, heads), (3, channels)):
                assert value_sizes[index] in (None, expected), value_sizes
        location_sizes = symbolic_helper._get_tensor_sizes(locations)
        queries = location_sizes[1] if location_sizes is not None else None
        result.setType(value.type().with_sizes([batch, queries, heads, channels]))
        shape = graph.op("Constant", value_t=torch.tensor([batch, -1, heads * channels], dtype=torch.int64))
        return graph.op("Reshape", result, shape).setType(value.type().with_sizes([batch, queries, heads * channels]))

    MSDeformAttnFunction.symbolic = staticmethod(symbolic)
