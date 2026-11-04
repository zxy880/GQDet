from __future__ import annotations

import json
from pathlib import Path
from PIL import Image


ROOT = None
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def yolo_to_coco_bbox(parts: list[str], width: int, height: int) -> tuple[int, list[float], float]:
    cls_id = int(float(parts[0]))
    x_center = float(parts[1]) * width
    y_center = float(parts[2]) * height
    box_w = float(parts[3]) * width
    box_h = float(parts[4]) * height

    x_min = max(0.0, x_center - box_w / 2.0)
    y_min = max(0.0, y_center - box_h / 2.0)
    box_w = min(box_w, width - x_min)
    box_h = min(box_h, height - y_min)
    area = max(0.0, box_w * box_h)
    return cls_id, [x_min, y_min, box_w, box_h], area


def build_coco_json(split: str) -> dict:
    images_dir = ROOT / "images" / split
    labels_dir = ROOT / "labels" / split

    image_id = 1
    ann_id = 1
    categories_seen: set[int] = set()
    coco = {"images": [], "annotations": [], "categories": []}

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO label in {label_path}: {raw_line}")
                cls_id, bbox, area = yolo_to_coco_bbox(parts, width, height)
                categories_seen.add(cls_id)
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls_id + 1,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        image_id += 1

    if not categories_seen:
        categories_seen.add(0)

    if categories_seen == {0}:
        coco["categories"] = [{"id": 1, "name": "ship", "supercategory": "ship"}]
    else:
        coco["categories"] = [
            {"id": cls_id + 1, "name": f"class_{cls_id}", "supercategory": "object"}
            for cls_id in sorted(categories_seen)
        ]

    return coco


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True, help='Existing images/{train,val,test} and labels/{train,val,test}')
    p.add_argument('--output-root', type=Path, required=True)
    args=p.parse_args()
    global ROOT
    ROOT=args.dataset_root
    args.output_root.mkdir(parents=True,exist_ok=True)
    for split in SPLITS:
        dest=args.output_root/f'instances_{split}.json'
        if dest.exists(): raise FileExistsError(dest)
        dest.write_text(json.dumps(build_coco_json(split),ensure_ascii=True),encoding='utf8')

if __name__ == '__main__': main()
