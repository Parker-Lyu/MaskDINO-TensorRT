"""Compare native preprocessing with Stage 2 on COCO and synthetic JPEGs."""
import argparse
import ctypes as ct
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from stage2.run import prepare

SIZE = 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    run = ct.CDLL(str(args.library.resolve())).stage4_preprocess
    run.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_void_p]

    ids = json.loads((Path(__file__).resolve().parents[1] / "stage2/sample.json").read_text())["smoke_image_ids"]
    images = json.loads((args.dataset / "annotations/panoptic_val2017.json").read_text())["images"]
    paths = {image["id"]: args.dataset / "val2017" / image["file_name"] for image in images}
    fixtures = [(str(image_id), paths[image_id].read_bytes()) for image_id in ids]
    rng = np.random.default_rng(42)
    for height, width, mode, orientation in (
        *((71, 137, "RGB", i) for i in range(1, 9)),
        (1101, 131, "RGB", 1), (16, 2048, "RGB", 1),
        (2048, 17, "RGB", 1), (1024, 1024, "RGB", 1),
        (69, 127, "L", 1), (73, 129, "CMYK", 8),
    ):
        bands = Image.getmodebands(mode)
        shape = (height, width) if bands == 1 else (height, width, bands)
        image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
        if mode == "CMYK":
            image = Image.frombytes(mode, (width, height), image.tobytes())
        exif = Image.Exif()
        exif[274] = orientation
        stream = BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        fixtures.append((f"{mode}-{height}x{width}-orientation-{orientation}", stream.getvalue()))

    mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)[:, None, None]
    std = np.array([58.395, 57.120, 57.375], dtype=np.float32)[:, None, None]
    failures = 0
    for name, jpeg in fixtures:
        with Image.open(BytesIO(jpeg)) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        canvas, valid, original = prepare(jpeg, SIZE)
        expected = (canvas.numpy() - mean) / std
        actual = np.empty_like(expected, order="C")
        decoded = np.empty_like(rgb)
        geometry = (ct.c_int * 4)()
        status = run(jpeg, len(jpeg), geometry, decoded.ctypes.data, actual.ctypes.data)
        assert status == 0, name
        decoded_delta = int(np.count_nonzero(rgb != decoded))
        delta = float(np.max(np.abs(expected - actual)))
        print(json.dumps(dict(image=name, geometry=list(geometry), decoded_differences=decoded_delta,
                              normalized_max_abs=delta)), flush=True)
        failures += tuple(geometry) != (*original, *valid) or decoded_delta > 0 or delta > 1e-7
    assert not failures, f"{failures} preprocessing mismatches"


if __name__ == "__main__":
    main()
