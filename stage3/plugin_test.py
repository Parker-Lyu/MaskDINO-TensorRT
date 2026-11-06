"""Validate NVIDIA MSDA v2 against the original CUDA op and grid_sample."""
import torch

from stage3 import plugin
from stage3.run import OUTPUT, dump
from stage3.trt_backend import make_plan


def validate():
    import tensorrt as trt
    from maskdino.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import (
        MSDeformAttnFunction, ms_deform_attn_core_pytorch,
    )

    class Op(torch.nn.Module):
        def forward(self, value, shapes, starts, locations, weights):
            return MSDeformAttnFunction.apply(value, shapes.long(), starts.long(), locations, weights, 64)

    plugin.install()
    torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = torch.tensor([[128, 128], [64, 64], [32, 32], [16, 16]], device="cuda", dtype=torch.int32)
    areas = shapes.prod(dim=1, dtype=torch.int32)
    starts = (areas.cumsum(0, dtype=torch.int32) - areas).contiguous()
    value = torch.randn(1, int(areas.sum()), 8, 32, device="cuda")
    locations = torch.rand(1, 300, 8, 4, 4, 2, device="cuda") * 1.5 - 0.25
    weights = torch.rand(1, 300, 8, 16, device="cuda").softmax(-1).reshape(1, 300, 8, 4, 4)
    inputs = (value, shapes, starts, locations, weights)
    names = ("value", "shapes", "starts", "locations", "weights")
    op = Op().cuda()
    native = op(*inputs)
    reference = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
    path = OUTPUT / "msda.onnx"
    torch.onnx.export(op, inputs, str(path), opset_version=17, dynamo=False, input_names=names, output_names=["result"])
    logger = trt.Logger(trt.Logger.INFO)
    plan = make_plan(path, logger)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    for name, tensor in zip(names, inputs):
        assert tuple(engine.get_tensor_shape(name)) == tuple(tensor.shape)
        assert engine.get_tensor_dtype(name) == (trt.int32 if name in ("shapes", "starts") else trt.float32)
        assert context.set_tensor_address(name, tensor.data_ptr())
    output = torch.empty_like(native)
    assert tuple(engine.get_tensor_shape("result")) == tuple(output.shape)
    assert context.set_tensor_address("result", output.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

    def errors(expected, actual):
        delta = actual - expected
        return {
            "max_abs": delta.abs().max().item(), "mean_abs": delta.abs().mean().item(),
            "relative_l2": (delta.norm() / expected.norm()).item(),
            "passed": torch.allclose(expected, actual, atol=1e-5, rtol=1e-4),
        }

    report = {
        "precision": "fp32", "seed": 20261010, "atol": 1e-5, "rtol": 1e-4,
        "inputs": {name: list(tensor.shape) for name, tensor in zip(names, inputs)},
        "spatial_shapes": shapes.tolist(), "starts": starts.tolist(),
        "cuda_vs_grid_sample": errors(reference, native), "trt_vs_cuda": errors(native, output),
    }
    dump("result.json", report)
    assert report["cuda_vs_grid_sample"]["passed"] and report["trt_vs_cuda"]["passed"], report
