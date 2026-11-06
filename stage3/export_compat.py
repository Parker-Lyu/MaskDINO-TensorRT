"""Keep legacy exporter constants on CPU for host-side shape inference.
Only serialized Constant attributes move; runtime model tensors remain CUDA.
"""
import torch
from torch.onnx._internal.torchscript_exporter import jit_utils

def install():
    original=jit_utils.GraphContext.op
    counts={'cuda_constant_attributes_moved_to_cpu':0}
    def op(self,name,*a,**kw):
        for key,value in kw.items():
            if isinstance(value,torch.Tensor) and value.is_cuda:
                kw[key]=value.detach().cpu()
                counts['cuda_constant_attributes_moved_to_cpu']+=1
        return original(self,name,*a,**kw)
    jit_utils.GraphContext.op=op
    return counts
