# GQDet — Original Modules and Research Materials

**License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).**

This release contains the authors' original GQDet modules and reproducibility materials for *Geometry-Preserving Representation and Quality-Coupled Query Training for SAR Ship Detection*. All original content distributed in this release is licensed under CC BY 4.0; see LICENSE and LICENSE_SCOPE.md.

## Contents

- `gqdet/gpdm.py`: the original GPDM feature-injection module.
- `gqdet/rgqs.py`: the original candidate selection, conflict resolution, quality/rank weighting and auxiliary schedule methods, separated from the detector criterion.
- `gqdet/qcqt_losses.py`: standalone tensor interfaces for QCC and the three RGQS auxiliary losses.
- `gqdet/ljo.py`: original matching-cost and negative-loss schedules.
- `configs/` and `settings/`: experiment-specific model/training parameters. HRSID uses 120 epochs; AIR-SARShip-2.0 uses 150 epochs.
- `datasets/`: exact per-image splits, original filename/ID mappings and integrity inventories.
- `tools/data/`: original preparation and annotation-conversion scripts.

## Scope and use

This is an original-module and research-material release, **not a standalone detector training distribution**. The third-party detector backbone, encoder/decoder, base criterion, matcher, trainer and training entry point are not bundled. The experiment configuration files document the integrated experiment; they cannot be passed unchanged to an unmodified upstream installation. Obtain D-FINE separately from [the official repository](https://github.com/Peterande/D-FINE), and follow its own license. PyTorch, torchvision and other external dependencies are installed separately and are not included in this archive.

See INTEGRATION.md for tensor interfaces and integration points. The GPDM implementation and extracted RGQS selection/schedule methods retain their experiment computations; the standalone loss interfaces are reorganized mathematical kernels. This release has not been retrained as a complete integrated detector and does not claim that the standalone package alone reproduces the reported AP values.

## Dataset preparation

Original dataset images and original/converted annotations are not distributed. Obtain the official data from their providers:

- HRSID: https://github.com/chaozhong2010/HRSID
- AIR-SARShip: https://radars.ac.cn/web/data/getData?dataType=SARship

Keep the provided image identities and splits unchanged. The conversion tools create labels locally from separately obtained originals. For example:

```sh
python tools/data/rebuild_annotations.py --dataset hrsid --hrsid-coco /path/to/original_annotations.json
python tools/data/rebuild_annotations.py --dataset air --air-xml-root /path/to/original_xml
```

Other preparation options are documented by each tool's `--help` and `datasets/逐图身份与划分说明.txt`. `datasets/annotation_checksums.json` contains checksums, not bounding-box labels.

## Attribution

Credit Hui Luo, Xinyang Zhang and Yang Du, cite GQDet and its Zenodo archive, link to CC BY 4.0, and indicate changes. Hui Luo and Xinyang Zhang contributed equally and share first authorship.

Repository: https://github.com/zxy880/GQDet

This release (v1.1.0): https://doi.org/10.5281/zenodo.23945298

Zenodo version family: https://doi.org/10.5281/zenodo.23910745

The CC BY grant applies to this release's original contents. It does not relicense separately obtained software, images, annotations or weights. Earlier releases and commits retain their own licensing scopes.
