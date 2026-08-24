# FLAME3 Job 2 Input-Modality Ablation

Generated: 2026-08-24T17:27:02+08:00

## Fixed Design

The four-channel stem is retained in every arm. RGB-only zeros the Thermal channel during both training and evaluation; Thermal-only zeros all three RGB channels during both training and evaluation. This keeps architecture, parameter count, initialization shape, and supervision fixed, so the only scientific variable is available modality information.

All other settings follow v1.1: current partial-label supervision, 640x512, physical batch 8, polynomial LR horizon 100, gradient clipping max_norm=5.0, AMP, ImageNet initialization, seeds 200/201/202, and 30 training epochs. Fusion reuses the historical v1.1 records and is not rerun. Comparisons use the arithmetic mean of epochs 26-30 per seed.

## Saved-Metric Results

- Fusion: complete seeds=3; mean S=0.7222770670524804; mean manual-val47 mIoU=0.7677696413836048.
- RGB-only: complete seeds=3; mean S=0.38323087687188967; mean manual-val47 mIoU=0.5231715512119907.
- Thermal-only: complete seeds=2; mean S=0.6385006915060479; mean manual-val47 mIoU=0.6603470694133643.

Thermal-only seed 201 failed with a non-finite gradient in the original run and reproduced the same failure under an exact-protocol retry. It is reported as missing and is not replaced. RGB-only seed 200's original 27-epoch artifact is preserved but excluded; the complete retry1 run is used.

No model inference was performed. This summary reads saved metrics and checkpoint metadata only; no test dataset or model prediction was read.
