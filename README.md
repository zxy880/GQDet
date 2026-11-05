# GQDet

## License: CC BY 4.0

The authors' original GQDet code and reproducibility materials are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE) for the full text and [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for scope and attribution. Retained third-party source portions keep their original terms and notices; see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

This package contains the GQDet model implementation, fixed per-image dataset splits, preprocessing/conversion tools, and final training settings. Third-party original and converted annotations are obtained or generated locally, not redistributed here.

## Contents

- `src/`: model, GPDM, QCC, RGQS, LJO and training infrastructure.
- `configs/gqdet_hrsid.yml`, `configs/gqdet_air.yml`: final dataset configurations.
- `settings/`: concise experimental settings.
- `datasets/`: original filenames, unique image identifiers, fixed splits and SHA256 inventories (no target-coordinate annotations).
- `tools/data/`: annotation conversion and image preparation.

## Environment

Use Python 3.11 and install `python -m pip install -r requirements.txt`. Exact versions validated locally are recorded in `requirements-validated.txt` (PyTorch CUDA build cu128). Select a CUDA build supported by your hardware.

## Data

Obtain images and annotations from the original providers, subject to their terms:

- HRSID: https://github.com/chaozhong2010/HRSID
- AIR-SARShip-2.0: https://radars.ac.cn/web/data/getData?newsColumnId=74fe223a-0b01-4830-8d99-1ba276e67ad8&pageType=en

Restore HRSID's exact split from its image filenames:

```sh
python tools/data/restore_hrsid_images.py --source-images /path/to/HRSID/images
```

Rebuild HRSID annotations from the complete compatible original COCO release (supply all necessary JSON files, not only the upstream training subset):

```sh
python tools/data/rebuild_annotations.py --dataset hrsid --hrsid-coco /path/to/original/train.json /path/to/original/test.json
```

The converter follows the experimental six-decimal YOLO intermediate representation before constructing COCO boxes. If you already have the original experimental YOLO labels separately, use `--hrsid-yolo-root /path/to/labels` instead of `--hrsid-coco`.

Rebuild AIR annotations from the official XML files using the fixed image-identity manifest, then prepare TIFF images with per-image 1st/99th percentile normalization:

```sh
python tools/data/rebuild_annotations.py --dataset air --air-xml-root /path/to/AIR-SARShip-2.0-xml
python tools/data/prepare_air_sarship20_official_input.py --image-root /path/to/AIR-SARShip-2.0-data --split-root datasets/air/source_tiff_split --output-root datasets/air
```

The reconstruction entry point never randomly resplits. Original and processed image identities are mapped in each dataset's `image_identity_manifest.csv` and `.json`. Output annotation paths in these manifests refer to files generated locally. `datasets/annotation_checksums.json` contains canonical-JSON hashes only, with no target coordinates; all three splits must match the experimental annotations before any output is written. A missing-image, instance-count or checksum mismatch indicates an incompatible/incomplete source annotation release and stops reconstruction. The upstream HRSID repository's single training JSON is not a complete replacement for the original experimental release; do not silently substitute versions. Local validation reproduced all six experimental COCO files from the original external HRSID YOLO labels and AIR XML files. Complete official-COCO reconstruction still requires the matching complete upstream release.

The lower-level `prepare_hrsid.py` (YOLO to COCO) and `prepare_air_sarship20.py` (XML to COCO) implementations are retained. Do not use the latter's random-split defaults to reproduce this paper; use `rebuild_annotations.py` and the archived memberships instead.

## Training

Run from this package root. Place the official D-FINE-S COCO pretrained file at `checkpoints/dfine_s_coco.pth`.

```sh
python train.py -c configs/gqdet_hrsid.yml --use-amp
python train.py -c configs/gqdet_air.yml --use-amp
```

| Setting | HRSID | AIR-SARShip-2.0 |
| --- | --- | --- |
| Input size | 640 × 640 | 640 × 640 |
| Training epochs | 120 | 150 |
| Batch size | 12 | 8 |
| Optimizer | AdamW | AdamW |

Both datasets use official D-FINE-S COCO initialization, AMP training, and EMA. Select the checkpoint by validation AP50:95. Common experimental settings are summarized in `settings/`; executable configurations are in `configs/`.

## Evaluation

Use a corresponding GQDet EMA weight file obtained separately:

```sh
python evaluate.py --dataset hrsid --checkpoint /path/to/gqdet_hrsid_ema.pth --output output/hrsid_test.json
python evaluate.py --dataset air --checkpoint /path/to/gqdet_air_ema.pth --output output/air_test.json
```

Evaluation uses FP32 EMA, 300 native predictions, COCO maxDets [1,10,100], and no additional confidence threshold or NMS. Reference Test AP50:95: HRSID 75.6076%; AIR 37.9960%.

Dataset images, third-party original/converted annotations and weight binaries are not included in this package. Upstream notices and licenses are retained. See `LICENSE_SCOPE.md` for the CC BY 4.0 grant and retained third-party terms.
