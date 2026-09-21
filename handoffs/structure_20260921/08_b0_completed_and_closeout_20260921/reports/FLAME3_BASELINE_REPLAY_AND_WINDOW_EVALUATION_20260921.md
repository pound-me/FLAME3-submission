# FLAME3 B0 replay and fixed-window evaluation

Date: 2026-09-21. Status: training and six-condition evaluation complete.

## Decision

The current-environment B0 replay completed seeds 200/201/202 for 30 epochs and
evaluated every retained epoch26-30 checkpoint under clean, thermal noise
0.02/0.05/0.10, thermal_zero and rgb_zero. All 15 checkpoint units completed.

The frozen historical recovery test failed for all 15 same-seed/same-epoch units.
Therefore the result status is:

`DESCRIPTIVE_ONLY_REPLAY_NOT_RECOVERED`

No historical claim is upgraded, no candidate is declared a formal robustness
improvement, and no additional candidate run is authorized by this result.

## Recovery result

- Required tolerance: strictly less than `1e-6` for clean Smoke IoU and No-Fire FP.
- Passing checkpoint units: `0/15`.
- Smoke IoU absolute differences: `0.0021516` to `0.0535206`.
- No-Fire FP absolute differences: `3.5400e-05` to `0.0028612`.
- Historical environment: Python 3.9.21, Torch 2.6.0+cu118, CUDA 11.8,
  cuDNN 90100.
- Replay environment: Python 3.11.5, Torch 2.1.0+cu121, CUDA 12.1,
  cuDNN 8801.
- `historical_environment_identical: false`.
- `formal_upgrading_allowed: false`.

## Replayed B0 window

| Seed | Clean Smoke IoU | Clean No-Fire FP | Smoke IoU drop at sigma=0.05 |
|---:|---:|---:|---:|
| 200 | 0.681498 | 0.000220 | 0.129230 |
| 201 | 0.703176 | 0.000048 | 0.144328 |
| 202 | 0.679139 | 0.000377 | 0.156018 |
| Mean | 0.687938 | 0.000215 | 0.143192 |

## Descriptive pairing only

The table below uses the new B0 replay for a same-seed descriptive comparison.
It is not a formal replacement for the unrecovered historical v1.1 baseline.
`Precision seeds` counts the frozen `+0.020` clean-Smoke threshold. `Guard seeds`
counts all clean and No-Fire guards. `Robust seeds` counts the frozen sigma=0.05
drop-reduction threshold. A formal arm would require the complete frozen rule,
not isolated seed counts.

| Arm | Mean clean Smoke delta | Precision seeds | Guard seeds | Robust seeds | Mean drop reduction |
|---|---:|---:|---:|---:|---:|
| S1 | -0.00738 | 0/3 | 1/3 | 1/3 | -0.01375 |
| S3 | -0.01155 | 0/3 | 1/3 | 0/3 | -0.04913 |
| S4 | -0.01044 | 0/3 | 1/3 | 1/3 | +0.01222 |
| S6 | -0.00383 | 0/3 | 2/3 | 0/3 | -0.05505 |
| R1 | -0.02975 | 0/3 | 0/3 | 3/3 | +0.09986 |
| R2 | -0.00343 | 0/3 | 2/3 | 0/3 | -0.07469 |
| R3 | -0.00964 | 0/3 | 2/3 | 3/3 | +0.10142 |
| B1 | -0.01647 | 0/3 | 1/3 | 1/3 | -0.02694 |
| E1 control | +0.01103 | 1/3 | 3/3 | 0/3 | -0.10544 |

R1 and R3 reproduce a strong descriptive thermal-noise pattern, but R1 fails
the clean guard in all seeds and R3 fails it in one seed. Neither reaches the
clean precision threshold in any seed. E1 has one positive precision seed but
does not meet the three-seed precision rule and becomes less robust at sigma=0.05.

Thus no evaluated research arm passes the frozen complete rule. Fire/Heat,
three-class mIoU and selection score S remain record-only.

## Integrity and boundaries

- Input hashes unchanged: true.
- Context hashes unchanged: true.
- Historical reports modified: false.
- test107 read: false.
- Prediction files saved: false.
- Training during evaluation: false.
- Extra candidate runs started: false.
- B0 checkpoints epoch26-30 retained for all three seeds.

The formal B0/E1 latency measurement remains pending a Windows idle interval.
The remote physical-disk backup is verified; the off-machine 8.18 GB archive copy
is still transferring and is not yet verified.

Primary machine-readable evidence: `BASELINE_WINDOW_REPORT.json`,
`EVALUATION_STATUS.json`, `QUEUE_STATUS.json`, and the three per-seed
`RESULT.json` records in the baseline replay result package.
