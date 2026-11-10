"""Generate raw FP32 TensorRT inputs with the native letterbox pipeline."""

import argparse
import ctypes as ct
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("image_ids", nargs="+", type=int)
    args = parser.parse_args()

    preprocess = ct.CDLL(str(Path("preprocess-check.so").resolve())).stage4_preprocess
    preprocess.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_void_p]
    preprocess.restype = ct.c_int
    tensor = np.empty((1, 3, 1024, 1024), dtype=np.float32)

    for image_id in args.image_ids:
        jpeg_path = args.dataset / "val2017" / f"{image_id:012d}.jpg"
        with Image.open(jpeg_path) as image:
            rgb = np.empty((image.height, image.width, 3), dtype=np.uint8)
        jpeg = jpeg_path.read_bytes()
        geometry = (ct.c_int * 4)()
        if preprocess(jpeg, len(jpeg), geometry, rgb.ctypes.data, tensor.ctypes.data):
            raise RuntimeError(f"Native preprocessing failed for COCO image {image_id}")
        tensor.tofile(f"{image_id}.f32")
        print(f"COCO {image_id}: original and valid geometry {list(geometry)}")


if __name__ == "__main__":
    main()
