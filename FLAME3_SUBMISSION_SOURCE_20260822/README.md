# FLAME3 submission source

This directory is the clean, auditable source of truth for the frozen PIDNet-S four-channel Fusion method with manual Smoke/Ignore supervision and global L2 gradient clipping (`max_norm=5.0`). It deliberately excludes failed fusion modules and historical launch wrappers.

## Frozen method

- Input: corrected RGB plus raw thermal JPG, concatenated into four channels.
- Network: PIDNet-S, unchanged P/I/D topology, `NUM_CLASSES=3`.
- Classes: Background, Smoke, Fire/Heat; ambiguous transition pixels use Ignore 255.
- Training-only annotations: A048-A150 manual Smoke/Ignore plus machine Fire/Heat.
- Development validation: A001-A047 manual three-class masks plus full val134 Fire/Heat metrics.
- Stability variable: global model-wide L2 gradient clipping at 5.0.
- Four-channel stem initialization: full Kaiming random initialization. The preregistered RGB-copy/IR-mean alternative was rejected because its mean Fire/Heat IoU delta was `-0.050595`, beyond the `-0.03` guard.
- Test policy: the 107-image test was evaluated once after method/checkpoint freeze; subsequent analyses are disclosed as post-hoc and are not used to retune the method.

## Frozen evidence summary

- FLAME2 zero-shot, 200 official test images: v1.1 minus baseline mean paired gains were `+0.250295` Smoke IoU, `+0.195014` Fire/Heat IoU and `+0.222654` S across three seeds. This is a coverage-mismatched transfer estimate/lower bound, not a number directly comparable with FLAME3.
- test107 post-hoc robustness: 29 perturbations and 18,618 per-image confusion rows; thermal additive noise was the earliest tested degradation pathway. This analysis is descriptive and was not used for method selection.
- Conv1 initialization ablation: the RGB-copy/IR-mean arm improved manual three-class mIoU by `+0.009761` on average with 2/3 positive seeds, but reduced Fire/Heat IoU by `-0.050595`; the frozen decision is `retain_random_four_channel_stem`.
- RTX 4090 efficiency at 640x512, batch 1, AMP, `augment=False`: 7,623,939 parameters, 7.471 GMACs, 14.942 GFLOPs under the 2xMAC convention, 7.185 ms mean latency, 7.716 ms P95, and 139.18 FPS.
- Complete data manifest: 734 unique samples (493 train, 134 validation, 107 test107) with per-file SHA256; manifest construction performs no training or inference.

## Package entry points

Run commands from this directory so Python resolves the formal packages without `sys.path.insert`.

Training:

```text
python -m src.train_baseline_v11 --config configs/pidnet_s_fusion_manual_smoke_v11_30e.yaml --root-dataset <bundle-root> --pretrained <PIDNet_S_ImageNet.pth.tar> --trainset reproducibility/splits/train.csv --validset reproducibility/splits/val.csv --manual-smoke-annotation-package <annotation-package> --epochs 30 --lr-total-epochs 100 --batch-size 8 --num-workers 4 --seed 200 --run-name <name> --device cuda:0 --amp
```

Frozen validation evaluation:

```text
python -m src.evaluate_flame3_manual_smoke_v2 --config configs/pidnet_s_fusion_manual_smoke_v11_30e.yaml --root-dataset <bundle-root> --val-csv reproducibility/splits/val.csv --annotation-package <annotation-package> --checkpoint <checkpoint> --output <result.json> --device cuda:0 --amp
```

Standard efficiency benchmark (`augment=False`, batch 1, 640x512, fixed warm-up):

```text
python -m src.benchmark_efficiency --config configs/pidnet_s_fusion_manual_smoke_v11_30e.yaml --checkpoint <checkpoint> --output <efficiency.json> --device cuda:0 --height 512 --width 640 --warmup 100 --iterations 200 --trials 10 --amp --data-root <bundle-root> --val-csv reproducibility/splits/val.csv --data-batches 100
```

Conv1 initialization ablation:

```text
python -m src.run_conv1_ablation_30e --submission-root <this-directory> --bundle-root <bundle-root> --pretrained <PIDNet_S_ImageNet.pth.tar> --train-csv reproducibility/splits/train.csv --val-csv reproducibility/splits/val.csv --annotation-package <annotation-package> --device cuda:0 --num-workers 4
```

FLAME2 zero-shot transfer (read-only; no training or threshold tuning):

```text
python -m src.evaluate_flame2_zero_shot --project-root <historical-project-support> --dataset-root <flame2-root> --output-dir <output-dir> --device cuda:0 --batch-size 8 --num-workers 0 --amp
```

Frozen test107 post-hoc strata and perturbation reproduction:

```text
python -m src.evaluate_test107_posthoc --project-root <historical-project-support> --bundle-root <bundle-root> --test-csv <frozen-test107-csv> --strata-manifest <strata-manifest.csv> --preregistration <posthoc-preregistration.json> --frozen-result <frozen-test-result.json> --output-dir <output-dir> --device cuda:0 --batch-size 4 --num-workers 0 --amp
```

The test107 entry point reproduces an already completed, disclosed post-hoc analysis. It must not be used for method selection, checkpoint selection, threshold tuning, or any change to method identity.

## Reproducibility records

- `reproducibility/splits/`: portable train/val/test membership and per-sample supervision choice.
- `reproducibility/split_v2_audit/`: temporal/spatial split rules, boundary buffer, and v1-to-v2 movement audit.
- `reproducibility/CONV1_INITIALIZATION_ABLATION_PREREGISTRATION.json`: frozen low-cost initialization ablation.
- `artifacts/conv1_initialization_ablation/runs/`: metrics, resolved config, environment and run summary for all six Conv1 runs; checkpoint binaries are intentionally excluded.
- `reproducibility/complete_split_file_manifest/`: per-sample RGB, thermal, TIFF, mask and boundary hashes for all frozen splits.
- `results/efficiency/`: standardized RTX 4090 efficiency measurement for the retained random-stem arm.
- `SUBMISSION_READINESS_REPORT.md` and `reproducibility/SUBMISSION_COMPLETION_AUDIT.json`: machine-checked final package audit.
- Every training run writes `resolved_config.json`, `environment.json`, source/split hashes, checkpoints, and epoch metrics.
- `src/build_reproducibility_manifest.py` hashes every referenced RGB, thermal, TIFF, mask, and boundary file on a machine holding the data.
- New checkpoints store the NumPy RNG state as primitive metadata plus an `int64` tensor, so exact resume remains compatible with PyTorch 2.1 restricted `weights_only=True` loading.

Run the final package audit from this directory:

```text
python -B -m src.audit_submission_package --submission-root .
```

Raw FLAME3/FLAME2 images and pretrained/checkpoint binaries are intentionally excluded. Their acquisition and redistribution remain governed by the original publishers.
