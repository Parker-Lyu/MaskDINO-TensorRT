import torch
from torch.onnx import symbolic_helper
from maskdino.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDeformAttnFunction

def install():
    def symbolic(g,value,shapes,starts,locations,weights,step):
        out=g.op('trt::MultiscaleDeformableAttnPlugin_TRT',value,g.op('Cast',shapes,to_i=6),g.op('Cast',starts,to_i=6),locations,weights,plugin_version_s='2',plugin_namespace_s='')
        sizes=symbolic_helper._get_tensor_sizes(value); ls=symbolic_helper._get_tensor_sizes(locations)
        out.setType(value.type().with_sizes([sizes[0],ls[1],sizes[2],sizes[3]]))
        out=g.op('Reshape',out,g.op('Constant',value_t=torch.tensor([1,-1,256],dtype=torch.int64)))
        return out.setType(value.type().with_sizes([sizes[0],ls[1],256]))
    MSDeformAttnFunction.symbolic=staticmethod(symbolic)
