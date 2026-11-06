"""Keep legacy exporter constant attributes on CPU for host shape inference."""
import torch
from torch.onnx._internal.torchscript_exporter import jit_utils


def install():
    original = jit_utils.GraphContext.op
    counts = {"cuda_constant_attributes_moved_to_cpu": 0}

    def op(context, name, *args, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                kwargs[key] = value.detach().cpu()
                counts["cuda_constant_attributes_moved_to_cpu"] += 1
        return original(context, name, *args, **kwargs)

    jit_utils.GraphContext.op = op
    return counts
