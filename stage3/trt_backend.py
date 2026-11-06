"""Export MaskDINO's learned network and run it on the current PyTorch stream."""
from collections import Counter
from pathlib import Path
import time

import torch

from stage2.run import pytorch_network
from stage3.run import OUTPUT, SIZE, digest, dump

PLUGIN = "MultiscaleDeformableAttnPlugin_TRT"
SHAPES = {
    "image": (1, 3, SIZE, SIZE),
    "logits": (1, 300, 133),
    "masks": (1, 300, SIZE // 4, SIZE // 4),
    "boxes": (1, 300, 4),
}


class Network(torch.nn.Module):
    def __init__(self, model, precision, mixed_scope):
        super().__init__()
        self.model = model
        self.precision = precision
        self.mixed_scope = mixed_scope

    def forward(self, image):
        if self.precision == "mixed" and self.mixed_scope == "backbone":
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                features = self.model.backbone(image)
            outputs, _ = self.model.sem_seg_head({name: value.float() for name, value in features.items()})
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.precision == "mixed" and self.mixed_scope == "all", cache_enabled=False):
                outputs = pytorch_network(self.model, image)
        return tuple(outputs[f"pred_{name}"].float() for name in ("logits", "masks", "boxes"))


def half_mask_projection(model):
    """Use FP16 only for the final mask dot product, retaining FP32 graph I/O."""
    from onnx import TensorProto, helper

    graph = model.graph
    projections = [node for node in graph.node if node.op_type == "Einsum" and "masks" in node.output]
    assert len(projections) == 1, "Expected the final mask projection to write masks directly"
    projection = projections[0]
    nodes = []
    for index, name in enumerate(projection.input):
        cast = f"stage3_mask_projection_input_{index}"
        nodes.append(helper.make_node("Cast", [name], [cast], to=TensorProto.FLOAT16))
    output = "stage3_mask_projection_output"
    nodes.append(helper.make_node("Einsum", [node.output[0] for node in nodes], [output],
                                  name=projection.name, equation="bqc,bchw->bqhw"))
    nodes.append(helper.make_node("Cast", [output], ["masks"], to=TensorProto.FLOAT))
    index = list(graph.node).index(projection)
    del graph.node[index]
    for offset, node in enumerate(nodes):
        graph.node.insert(index + offset, node)


def export(model, image, precision, cpu_constants, mixed_scope):
    import onnx
    from torch.onnx._internal.torchscript_exporter import jit_utils
    from stage3 import export_compat, plugin

    assert precision in ("fp32", "mixed")
    original = jit_utils.GraphContext.op
    counts = export_compat.install() if cpu_constants else {}
    plugin.install()
    destination = OUTPUT / "network.onnx"
    try:
        torch.onnx.export(
            Network(model, precision, mixed_scope).eval(), image, str(destination), opset_version=17, dynamo=False,
            input_names=["image"], output_names=["logits", "masks", "boxes"],
        )
    finally:
        jit_utils.GraphContext.op = original
    graph = onnx.load(destination)
    if precision == "mixed" and mixed_scope == "mask-projection":
        half_mask_projection(graph)
        onnx.save(graph, destination)
    onnx.checker.check_model(graph)
    nodes = [node for node in graph.graph.node if node.op_type == PLUGIN]
    assert len(nodes) == 15
    for node in nodes:
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        assert node.domain == "trt" and len(node.input) == 5
        assert attrs == {"plugin_version": b"2", "plugin_namespace": b""}
    dump("export.json", {
        "onnx": onnx.__version__, "torch": torch.__version__, "precision": precision,
        "opset": 17, "dynamo": False, "sha256": digest(destination),
        "mixed_scope": mixed_scope if precision == "mixed" else None,
        "operators": dict(Counter(f"{node.domain}::{node.op_type}" for node in graph.graph.node)),
        "plugin_count": len(nodes), "plugin_version": "2", **counts,
    })


def make_plan(path, logger):
    import tensorrt as trt

    trt.init_libnvinfer_plugins(logger, "")
    assert trt.get_plugin_registry().get_creator(PLUGIN, "2", "") is not None
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return plan


def build(path, precision):
    import json
    import tensorrt as trt

    assert precision in ("fp32", "mixed")
    start = time.perf_counter()
    plan = make_plan(path, trt.Logger(trt.Logger.INFO))
    destination = OUTPUT / "network.engine"
    destination.write_bytes(bytes(plan))
    engine = Engine(destination)
    inspector = engine.engine.create_engine_inspector()
    inspector.execution_context = engine.context
    layers = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    (OUTPUT / "layers.json").write_text(layers)
    plugin_layers = [layer for layer in json.loads(layers)["Layers"] if "Plugin" in layer["LayerType"]]
    assert len(plugin_layers) == 15
    output = engine(torch.zeros(SHAPES["image"], device="cuda"))
    assert all(torch.isfinite(tensor).all().item() for tensor in output.values())
    if precision == "fp32":
        assert all(tensor.dtype == torch.float32 for tensor in output.values())
    dump("build.json", {
        "onnx_sha256": digest(path), "engine_sha256": digest(destination),
        "tensorrt": trt.__version__, "precision": precision, "seconds": time.perf_counter() - start,
        "strongly_typed": True, "tf32": False, "max_aux_streams": 0, "workspace_bytes": 4 * 1024**3,
        "plugin_count": len(plugin_layers), "finite_smoke": True,
        "io": {name: {"shape": shape, "dtype": str(engine.engine.get_tensor_dtype(name))}
               for name, shape in SHAPES.items()},
    })


class Engine:
    """Own one context and one set of output buffers for serial inference."""

    def __init__(self, path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {path}")
        self.context = self.engine.create_execution_context()
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        assert names == SHAPES.keys()
        assert self.engine.get_tensor_dtype("image") == trt.float32
        self.outputs = {}
        for name, shape in SHAPES.items():
            assert tuple(self.context.get_tensor_shape(name)) == shape
            mode = trt.TensorIOMode.INPUT if name == "image" else trt.TensorIOMode.OUTPUT
            assert self.engine.get_tensor_mode(name) == mode
            if mode == trt.TensorIOMode.OUTPUT:
                dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[self.engine.get_tensor_dtype(name)]
                output = torch.empty(shape, device="cuda", dtype=dtype)
                assert self.context.set_tensor_address(name, output.data_ptr())
                self.outputs[f"pred_{name}"] = output

    def __call__(self, image):
        assert image.is_cuda and image.is_contiguous() and image.dtype == torch.float32
        assert tuple(image.shape) == SHAPES["image"]
        assert self.context.set_tensor_address("image", image.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.outputs
