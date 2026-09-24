# FLAME3 B0-S7 Matched 100-Epoch Confirmation Report

Date: 2026-09-24
Status: B0_S7_100E_ENDED_STOP_PENDING_NEW_AUTHORIZATION
Stop reason: Authorized matched B0-S7 100-epoch confirmation ended. No later stage was started.

## Execution

- Formal runs started/completed/evaluated: 6/6/6; authorized maximum 6.
- Failure events in repair run: 1. Resume events: 2.
- Preserved prior failed attempts: 1; these are not erased from the experiment registry.
- Models: matched B0 and exact Stage A S7; seeds 203/204/205; 100 epochs each; poly100.
- Main endpoint window: epochs 96-100. Epochs 26-30 are trajectory-only and were not used for checkpoint selection.
- This remains a repeatedly used development-set experiment; new seeds are not a new independent dataset.

## Overall Decision

- Status: PASS_ROBUST_AXIS_ONLY.
- Precision axis: False. Robust axis: True.
- Mean clean delta: +0.4654 pp.
- Mean sigma=.05 absolute delta: +7.8851 pp.
- Mean degradation improvement R: +7.4197 pp.
- Thresholds are project utility gates, not statistical significance claims.

## Same-Seed Paired Results

| Seed | B0 clean (%) | S7 clean (%) | Delta clean (pp) | Delta noise .05 (pp) | R (pp) | Clean guard | FP abs | FP relative | Precision pass | Robust pass |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| 203 | 71.7403 | 73.3347 | +1.5944 | +4.6946 | +3.1002 | True | True | True | False | True |
| 204 | 73.0162 | 72.1116 | -0.9046 | +11.1422 | +12.0468 | True | True | True | False | True |
| 205 | 72.9811 | 73.6876 | +0.7065 | +7.8186 | +7.1121 | True | True | True | False | True |

## Modality-Zero Side Effects

| Seed | S7 thermal-zero FP (%) | Delta vs B0 (pp) | S7 rgb-zero Smoke IoU (%) | Delta vs B0 (pp) |
|---:|---:|---:|---:|---:|
| 203 | 3.7270 | +2.9547 | 5.1014 | +5.0316 |
| 204 | 0.7391 | +0.5630 | 4.3498 | -5.1605 |
| 205 | 0.3636 | +0.2934 | 2.5609 | -15.1968 |

## Evidence and Boundary

- Fire/Heat, Background, mIoU and S are record-only.
- No best-checkpoint, per-seed model choice, early stopping or threshold change was used.
- No test107 image, label, prediction, statistic or sample list was accessed.
- Original Stage A and Stage B decisions remain unchanged.
- Next stage approved: NO. Further training requires a new explicit authorization.
