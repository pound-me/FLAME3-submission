# FLAME3 frozen checkpoints on FLAME2: zero-shot transfer

> Read-only inference on the official 200-image FLAME2 test split. No training, fine-tuning, threshold selection, or prediction saving.

## Frozen protocol

- FLAME2 mapping: 0=Background, 1=Smoke, 2=Fire/Heat.
- Label audit: FLAME2 provides an independent Smoke class, and class-2 regions were observed under dense RGB smoke while remaining aligned with thermal hotspots; class 2 is therefore not restricted to RGB-visible flame.
- No synthetic Ignore pixels were introduced; the measured Ignore ratio is reported explicitly below.
- IR conversion: official loader `PIL.convert('L')`, then division by 255.
- Spatial input: deterministic 256x256 official FLAME2 preprocessing.
- Model inference: `augment=False`; six SHA256-pinned best_S checkpoints.
- Coverage warning: FLAME2 Fire cannot be proven to cover every residual-heat footprint in the broader FLAME3 class; treat the numbers as a coverage-mismatched transfer estimate/lower bound.

## Three-seed aggregate

| Arm | Smoke IoU | Smoke precision | Smoke recall | Fire/Heat IoU | Fire/Heat precision | Fire/Heat recall |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.010928 +/- 0.018928 | 0.049680 +/- 0.086048 | 0.013444 +/- 0.023287 | 0.087175 +/- 0.038777 | 0.518948 +/- 0.097785 | 0.095378 +/- 0.043180 |
| v11 | 0.261223 +/- 0.078317 | 0.891777 +/- 0.081085 | 0.272485 +/- 0.087644 | 0.282189 +/- 0.049455 | 0.616195 +/- 0.047649 | 0.347721 +/- 0.086214 |

| Arm | 3-class mIoU | Equal Fire-Smoke S | BG-only joint FP | Ignore pixel ratio |
|---|---:|---:|---:|---:|
| baseline | 0.245154 +/- 0.013061 | 0.049051 +/- 0.023165 | 0.092491 +/- 0.135968 | 0.000000 +/- 0.000000 |
| v11 | 0.423497 +/- 0.030586 | 0.271706 +/- 0.034224 | 0.030410 +/- 0.034228 | 0.000000 +/- 0.000000 |

## Paired v1.1 minus baseline

| Metric | Mean paired delta | Seeds improved in desired direction |
|---|---:|---:|
| smoke_iou | +0.250295 | 3/3 |
| smoke_precision | +0.842097 | 3/3 |
| smoke_recall | +0.259040 | 3/3 |
| fire_heat_iou | +0.195014 | 3/3 |
| fire_heat_precision | +0.097246 | 2/3 |
| fire_heat_recall | +0.252342 | 3/3 |
| three_class_mean_iou | +0.178343 | 3/3 |
| equal_fire_smoke_S | +0.222654 | 3/3 |
| background_only_joint_false_positive_ratio | -0.062081 | 2/3 |
| ignore_pixel_ratio | +0.000000 | n/a (unchanged) |

For the false-positive ratio, lower is better. Ignore remains zero because the audited FLAME2 release supplies dense three-class masks and the frozen mapping introduces no synthetic Ignore pixels.

This experiment is cross-dataset evidence only and must not be numerically merged with the FLAME3 validation or test tables.
