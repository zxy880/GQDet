"""Prepare AIR-SARShip-2.0 for an unmodified official D-FINE checkout.

The official custom COCO loader expects zero-based category IDs. Native 16-bit
TIFF images are converted to lossless 8-bit PNG after robust per-image 1st/99th
percentile normalization, so the official PIL/RGB loader can be used unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_images = args.output_root / "images"
    output_annotations = args.output_root / "annotations"
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)

    referenced_names: set[str] = set()
    summary: dict[str, object] = {"splits": {}, "category_id": 0}
    for split in ("train", "val", "test"):
        source_json = args.split_root / "annotations" / f"instances_{split}.json"
        coco = json.loads(source_json.read_text(encoding="utf-8"))
        for image in coco["images"]:
            source_name = image["file_name"]
            referenced_names.add(source_name)
            image["file_name"] = f"{Path(source_name).stem}.png"
        for annotation in coco["annotations"]:
            annotation["category_id"] = 0
        coco["categories"] = [{"id": 0, "name": "ship", "supercategory": "ship"}]
        (output_annotations / f"instances_{split}.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["splits"][split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }

    for source_name in sorted(referenced_names, key=lambda name: int(Path(name).stem)):
        source_path = args.image_root / source_name
        target_path = output_images / f"{source_path.stem}.png"
        with Image.open(source_path) as image:
            pixels = np.asarray(image)
        if pixels.dtype != np.uint16:
            raise RuntimeError(f"Expected uint16 TIFF, got {pixels.dtype}: {source_path}")
        low, high = np.percentile(pixels, [1.0, 99.0])
        if high <= low:
            converted = np.zeros_like(pixels, dtype=np.uint8)
        else:
            converted = np.clip(
                (pixels.astype(np.float32) - low) * (255.0 / (high - low)),
                0,
                255,
            ).astype(np.uint8)
        Image.fromarray(converted).save(target_path, format="PNG", optimize=True)

    summary["converted_images"] = len(referenced_names)
    (args.output_root / "preparation_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
