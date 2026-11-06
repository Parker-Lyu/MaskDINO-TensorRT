"""Verify both interpolation steps and the crop against PyTorch CUDA."""
import argparse
import ctypes as ct
import json

import torch
from torch.nn import functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess


class Geometry(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ("height", "width", "valid_height", "valid_width")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    args = parser.parse_args()
    run = ct.CDLL(args.library).stage4_restore
    run.argtypes = [ct.c_void_p, ct.c_void_p, Geometry, ct.c_int, ct.c_void_p]
    torch.manual_seed(42)
    masks = torch.randn(8, 256, 256, device="cuda")
    square = F.interpolate(masks[None], (1024, 1024), mode="bilinear", align_corners=False)[0]
    stream = torch.cuda.current_stream().cuda_stream
    for h, w in ((1024, 1024), (385, 640), (640, 427), (512, 2048), (1339, 701), (1, 512), (1, 1)):
        scale = 1024 / max(h, w)
        valid_h, valid_w = max(1, round(h * scale)), max(1, round(w * scale))
        expected = sem_seg_postprocess(square, (valid_h, valid_w), h, w)
        actual = torch.empty_like(expected, memory_format=torch.contiguous_format)
        assert run(masks.data_ptr(), actual.data_ptr(), Geometry(h, w, valid_h, valid_w), len(masks), stream) == 0
        delta = (expected - actual).abs()
        print(json.dumps(dict(geometry=[h, w, valid_h, valid_w], max_abs=delta.max().item(),
                              mean_abs=delta.mean().item())), flush=True)
        torch.testing.assert_close(expected, actual, atol=2e-6, rtol=1e-6)


if __name__ == "__main__":
    main()
