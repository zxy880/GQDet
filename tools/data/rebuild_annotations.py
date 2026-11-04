"""Rebuild annotations locally from separately obtained original labels; never resplit."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from prepare_air_sarship20 import parse_boxes
from prepare_hrsid import yolo_to_coco_bbox

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["hrsid", "air"], required=True)
    p.add_argument("--hrsid-yolo-root", type=Path, help="External original experimental YOLO labels, flat or recursive")
    p.add_argument("--hrsid-coco", type=Path, nargs="+", help="Complete compatible official COCO annotation JSON file(s)")
    p.add_argument("--air-xml-root", type=Path, help="Official AIR XML directory")
    p.add_argument("--output-root", type=Path)
    args = p.parse_args()
    base = ROOT / "datasets" / args.dataset
    rows = json.loads((base / "image_identity_manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "datasets/annotation_checksums.json").read_text())
    out = args.output_root or base
    sources = {}
    if args.dataset == "hrsid":
        if bool(args.hrsid_yolo_root) == bool(args.hrsid_coco):
            p.error("Specify exactly one of --hrsid-yolo-root or --hrsid-coco")
        if args.hrsid_yolo_root:
            for f in args.hrsid_yolo_root.rglob("*.txt"):
                if f.stem in sources and sources[f.stem].read_bytes() != f.read_bytes():
                    raise ValueError(f"Conflicting labels: {f.stem}")
                sources[f.stem] = f
        else:
            for f in args.hrsid_coco:
                data = json.loads(f.read_text(encoding="utf-8"))
                boxes = defaultdict(list)
                for a in data["annotations"]:
                    boxes[a["image_id"]].append(a["bbox"])
                for im in data["images"]:
                    key = Path(im["file_name"]).stem
                    lines = []
                    for x, y, w, h in boxes[im["id"]]:
                        lines.append("0 " + " ".join(f"{v:.6f}" for v in ((x+w/2)/im["width"], (y+h/2)/im["height"], w/im["width"], h/im["height"])))
                    if key in sources and sources[key] != lines:
                        raise ValueError(f"Conflicting COCO annotations: {key}")
                    sources[key] = lines
    elif not args.air_xml_root:
        p.error("--air-xml-root is required for AIR")

    products = {}
    for split in ("train", "val", "test"):
        selected = sorted((r for r in rows if r["split"] == split), key=lambda r: r["coco_image_id"])
        coco = {"images": [], "annotations": [], "categories": [{"id": 1 if args.dataset == "hrsid" else 0, "name": "ship", "supercategory": "ship"}]}
        if args.dataset == "air":
            coco.update(info={"description": "AIR-SARShip-2.0 single-class ship detection"}, licenses=[])
        for r in selected:
            w, h, image_id = r["width"], r["height"], r["coco_image_id"]
            coco["images"].append({"id": image_id, "file_name": r["training_file_name"], "width": w, "height": h})
            key = Path(r["original_file_name"]).stem
            if args.dataset == "air":
                boxes, _ = parse_boxes(args.air_xml_root / (key + ".xml"), w, h)
                values = [(0, b, b[2]*b[3]) for b in boxes]
            else:
                if key not in sources:
                    raise ValueError(f"Missing original annotation: {key}")
                lines = sources[key].read_text().splitlines() if isinstance(sources[key], Path) else sources[key]
                values = []
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(f"Invalid YOLO label: {key}")
                    cls, box, area = yolo_to_coco_bbox(parts, w, h)
                    if cls != 0:
                        raise ValueError(f"Expected single ship class: {key}")
                    values.append((1, box, area))
            if len(values) != r["instance_count"]:
                raise ValueError(f"Annotation version/count mismatch: {key}")
            for cat, box, area in values:
                coco["annotations"].append({"id": len(coco["annotations"])+1, "image_id": image_id, "category_id": cat, "bbox": box, "area": area, "iscrowd": 0})
        name = f"annotations/instances_{split}.json"
        if digest(coco) != expected[args.dataset][split]:
            raise ValueError(f"{args.dataset}/{split}: source annotations differ from experimental version. Obtain the compatible original release; do not silently substitute annotations.")
        products[name] = coco
        if args.dataset == "air":
            source = json.loads(json.dumps(coco))
            for im, r in zip(source["images"], selected):
                im["file_name"] = r["original_file_name"]
            source["categories"][0]["id"] = 1
            for a in source["annotations"]:
                a["category_id"] = 1
            products[f"source_tiff_split/annotations/instances_{split}.json"] = source
    # Check all sources before writing any outputs.
    for name, data in products.items():
        dest = out / name
        if dest.exists() and json.loads(dest.read_text()) != data:
            raise FileExistsError(f"Conflicting output: {dest}")
    for name, data in products.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "images": len(rows), "verified": True, "files": list(products)}))


if __name__ == "__main__":
    main()
