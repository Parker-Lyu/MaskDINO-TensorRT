"""Lower the upstream MSDA autograd op to NVIDIA TensorRT MSDA v2."""
import torch
from torch.onnx import symbolic_helper

from maskdino.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDeformAttnFunction


def install():
    def symbolic(graph, value, shapes, starts, locations, weights, _step):
        result = graph.op(
            "trt::MultiscaleDeformableAttnPlugin_TRT",
            value, graph.op("Cast", shapes, to_i=6), graph.op("Cast", starts, to_i=6),
            locations, weights, plugin_version_s="2", plugin_namespace_s="",
        )
        batch, _, heads, channels = symbolic_helper._get_tensor_sizes(value)
        queries = symbolic_helper._get_tensor_sizes(locations)[1]
        result.setType(value.type().with_sizes([batch, queries, heads, channels]))
        shape = graph.op("Constant", value_t=torch.tensor([batch, -1, heads * channels], dtype=torch.int64))
        return graph.op("Reshape", result, shape).setType(value.type().with_sizes([batch, queries, heads * channels]))

    MSDeformAttnFunction.symbolic = staticmethod(symbolic)
