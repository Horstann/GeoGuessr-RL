#!/usr/bin/env python3
"""Resize images recursively, preserving filenames and relative directories.

Choose a maximum pixel count directly or through a zoom-level pixel budget.
Smaller images are copied unchanged. Larger images retain their aspect ratio
within whole-pixel rounding, with a minimum of one pixel per side.
Requires Pillow (pip install Pillow).
"""

import argparse
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ZOOM_SIZE_MAP = {
    # zoom_level: (width, height, total_pixels)
    0: (512, 256, 131_072),
    1: (1_024, 512, 524_288),
    2: (2_048, 1_024, 2_097_152),
    3: (4_096, 2_048, 8_388_608),
    4: (8_192, 4_096, 33_554_432),
    5: (13_312, 6_656, 88_604_672),
}
ZOOM_LEVELS = list(ZOOM_SIZE_MAP.keys())

def parse_zoom(value):
    value = int(value)
    assert value in ZOOM_LEVELS, f"Got invalid zoom {value}"
    return value
def parse_max_pixels(value):
    value = int(value)
    assert 0 < value, f"Got invalid max_pixels {value}"
    return value
def parse_bool(value):
    if value.lower() in {"true", "yes", "1"}:
        return True
    if value.lower() in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")

def resize_image(task):
    src, dst, max_pixels = task
    try:
        with Image.open(src) as img:
            dst.parent.mkdir(parents=True, exist_ok=True)
            num_pixels = img.size[0] * img.size[1]
            scale = (max_pixels / num_pixels) ** 0.5
            if scale < 1:
                width, height = (max(1, math.floor(side * scale)) for side in img.size)
                with img.resize((width, height), resample=Image.Resampling.LANCZOS) as resized:
                    resized.save(dst)
            else:
                shutil.copy(src, dst)
        return None
    except Exception as exc:
        return f"{src}: {exc}"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", type=Path, default="data/pano_images_test")
    parser.add_argument("--output-folder", type=Path, default="data/pano_images_resized")
    parser.add_argument(
        "--multiprocess", type=parse_bool, nargs="?", const=True, default=False,
        help="Use multiple processes (flag alone or true/false; default: false)",
    )
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument(
        "--zoom", type=parse_zoom, required=False, default=None,
        help="Discrete zoom levels (0,1,2,3,4,5) corresponding to Google Street View Tiles",
    )
    budget.add_argument(
        "--max-pixels", type=parse_max_pixels, required=False, default=None,
        help="Maximum no. of pixels allowable",
    )
    
    args = parser.parse_args()
    input_folder = args.input_folder.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()
    if not input_folder.is_dir():
        parser.error(f"Input folder does not exist: {input_folder}")
    if output_folder == input_folder or output_folder in input_folder.parents:
        parser.error("Output folder must not equal or contain the input folder")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Both dimensions scale by sqrt(remaining pixel fraction).
    if args.zoom is not None:
        max_pixels = ZOOM_SIZE_MAP[args.zoom][-1]
    else:
        max_pixels = args.max_pixels
    extensions = Image.registered_extensions()
    tasks = [
        (path, output_folder / path.relative_to(input_folder), max_pixels)
        for path in sorted(input_folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
        and output_folder not in path.parents
    ]
    if not tasks:
        print("No image files found.")
        return 0

    failures = 0
    if args.multiprocess:
        with ProcessPoolExecutor() as executor:
            for error in executor.map(resize_image, tasks):
                if error:
                    print(error, file=sys.stderr)
                    failures += 1
    else:
        for error in tqdm(
            map(resize_image, tasks),
            total=len(tasks),
            desc="Resizing images",
        ):
            if error:
                print(error, file=sys.stderr)
                failures += 1
    print(f"Resized {len(tasks) - failures}/{len(tasks)} images into {output_folder}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
