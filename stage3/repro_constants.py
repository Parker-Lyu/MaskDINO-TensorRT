"""Exercise exporter host shape inference with CUDA Slice index constants.
No MaskDINO, Detectron2, custom kernels, model weights or TensorRT involved.
"""
import sys,torch
from torch.onnx._internal.torchscript_exporter import jit_utils
from stage3.export_compat import install
GLOBALS=jit_utils.GLOBALS
counts=install() if '--cpu-constants' in sys.argv else {}
GLOBALS.export_onnx_opset_version=17
GLOBALS.onnx_shape_inference=True
graph=torch._C.Graph()
x=graph.addInput('mask')
x.setType(torch._C.TensorType.create_from_tensor(torch.zeros((1,8500),dtype=torch.bool)).with_sizes([None,None]))
ctx=jit_utils.GraphContext(graph,graph.block(),17,None,{}, {},set())
start=ctx.op('Constant',value_t=torch.tensor([100],device='cuda',dtype=torch.int64))
end=ctx.op('Constant',value_t=torch.tensor([500],device='cuda',dtype=torch.int64))
axis=ctx.op('Constant',value_t=torch.tensor([1],dtype=torch.int64))
step=ctx.op('Constant',value_t=torch.tensor([1],dtype=torch.int64))
print('BEFORE_SLICE',flush=True)
y=ctx.op('Slice',x,start,end,axis,step)
graph.registerOutput(y)
print('AFTER_SLICE',str(graph),counts,flush=True)
