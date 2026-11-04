"""Convert AIR-SARShip-2.0 XML rectangles to a reproducible COCO split.

The public release has no official train/test split.  Split sizes and the
random seed are explicit command-line arguments.  Every XML category is
collapsed to the single detection class ``ship`` to match D-FINE-S.
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


def numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def parse_boxes(xml_path: Path, width: int, height: int) -> tuple[list[list[float]], Counter]:
    root = ET.parse(xml_path).getroot()
    boxes: list[list[float]] = []
    labels: Counter = Counter()
    for obj in root.findall("./objects/object"):
        name = (obj.findtext("./possibleresult/name") or "ship").strip()
        points = []
        for point in obj.findall("./points/point"):
            x, y = (float(v.strip()) for v in point.text.split(","))
            points.append((x, y))
        if len(points) < 2:
            continue
        x0 = max(0.0, min(x for x, _ in points))
        y0 = max(0.0, min(y for _, y in points))
        x1 = min(float(width), max(x for x, _ in points))
        y1 = min(float(height), max(y for _, y in points))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append([x0, y0, x1 - x0, y1 - y0])
        labels[name] += 1
    return boxes, labels


def build_coco(image_paths: list[Path], xml_dir: Path) -> tuple[dict, Counter]:
    images, annotations = [], []
    annotation_id = 1
    source_labels: Counter = Counter()
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        boxes, labels = parse_boxes(xml_dir / f"{image_path.stem}.xml", width, height)
        source_labels.update(labels)
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        for x, y, w, h in boxes:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {"description": "AIR-SARShip-2.0 single-class ship detection"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ship", "supercategory": "ship"}],
    }, source_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--train-count", type=int, default=189)
    parser.add_argument("--val-count", type=int, default=21)
    parser.add_argument("--test-count", type=int, default=90)
    args = parser.parse_args()

    image_dir = args.raw_root / "AIR-SARShip-2.0-data"
    xml_dir = args.raw_root / "AIR-SARShip-2.0-xml"
    images = sorted(image_dir.glob("*.tiff"), key=numeric_key)
    if len(images) != 300:
        raise RuntimeError(f"Expected 300 TIFF images, found {len(images)}")
    requested_total = args.train_count + args.val_count + args.test_count
    if min(args.train_count, args.val_count, args.test_count) <= 0:
        raise RuntimeError("Every split must contain at least one image")
    if requested_total != len(images):
        raise RuntimeError(
            f"Split counts total {requested_total}, but found {len(images)} images"
        )
    missing_xml = [image.stem for image in images if not (xml_dir / f"{image.stem}.xml").is_file()]
    if missing_xml:
        raise RuntimeError(f"Missing XML annotations for: {missing_xml[:10]}")

    shuffled = images.copy()
    random.Random(args.seed).shuffle(shuffled)
    splits = {
        "test": sorted(shuffled[: args.test_count], key=numeric_key),
        "val": sorted(
            shuffled[args.test_count : args.test_count + args.val_count],
            key=numeric_key,
        ),
        "train": sorted(
            shuffled[args.test_count + args.val_count :], key=numeric_key
        ),
    }
    split_names = {name: {path.name for path in paths} for name, paths in splits.items()}
    if (
        split_names["train"] & split_names["val"]
        or split_names["train"] & split_names["test"]
        or split_names["val"] & split_names["test"]
    ):
        raise RuntimeError("Split overlap detected before annotation conversion")

    annotation_dir = args.output_root / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "split_sizes": {},
        "source_labels": Counter(),
        "split_overlap": False,
    }
    for split, split_images in splits.items():
        coco, labels = build_coco(split_images, xml_dir)
        (annotation_dir / f"instances_{split}.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["split_sizes"][split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }
        summary["source_labels"].update(labels)
        (args.output_root / f"{split}.txt").write_text(
            "\n".join(image.name for image in split_images) + "\n", encoding="utf-8"
        )
    (args.output_root / "split_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
