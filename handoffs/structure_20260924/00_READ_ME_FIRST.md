# FLAME3: 2026-09-24 handoff for web review

This is a status and evidence index, not a new experiment order. Read the linked
reports before interpreting any score. The 2026-09-21 `CURRENT_STATUS.md` is an
older snapshot and must not be treated as today's status.

## Current position (2026-09-24, Asia/Shanghai)

1. On the RTX4060Ti, the original 4090 baseline could not be formally carried
   over: all three 30-epoch seeds completed, but the frozen cross-machine
   acceptance gate failed. The same-checkpoint replay was consistent; neither
   independent same-machine training repeatability nor cross-machine equivalence
   follows from that result. The R1 four-point parameter sweep was **not run**.
   [Read-only diagnosis](reports/FLAME3_4060_BASELINE_CARRYOVER_READONLY_DIAGNOSIS_20260922.md).
2. A *separate, internally paired 4060Ti comparison* completed the Stage A
   second batch (S2/S5/S7/S8). S7 was `PROMISING_NOT_PASS` at 30 epochs;
   S8 and S5 did not pass; S2 was blocked by engineering and has no formal
   segmentation result. Do not combine this group's scores with the 4090
   leaderboard. [Stage A report](reports/FLAME3_4060_SECOND_BATCH_REPORT_RECOVERY03_20260922.md).
3. The limited Stage B S8-G10/G80 follow-up found no passing candidate. S8-G20
   retains its earlier non-pass status; do not resume an unrestricted grid
   search from these observations. [Stage B analysis](reports/FLAME3_S8_STAGEB_ANALYSIS_20260923.md).
4. The authorized B0/S7 100-epoch matched confirmation is **finished and
   stopped**. Six runs, seeds 203/204/205, all completed and evaluated. Fixed
   primary window: epochs 96-100; no best-checkpoint selection. The decision
   is `PASS_ROBUST_AXIS_ONLY`: robustness passed in 3/3 seeds, precision in
   0/3. The project has not authorized another training stage.
   [Final report](reports/B0_S7_100E_CONFIRMATION_REPORT_20260923.md),
   [frozen decision JSON](evidence/DECISIONS.json),
   [paired deltas](evidence/PAIRED_DELTAS.csv),
   [completion check](evidence/FINAL_COMPLETION_CHECK.json).

## Matched 100-epoch B0/S7 result

| Seed | Clean Smoke IoU: B0 -> S7 | Delta clean | Delta noise sigma=.05 | Reduction in degradation R | Precision | Robustness |
|---:|---:|---:|---:|---:|---|---|
| 203 | 71.7403% -> 73.3347% | +1.5944 pp | +4.6946 pp | +3.1002 pp | Fail | Pass |
| 204 | 73.0162% -> 72.1116% | -0.9046 pp | +11.1422 pp | +12.0468 pp | Fail | Pass |
| 205 | 72.9811% -> 73.6876% | +0.7065 pp | +7.8186 pp | +7.1121 pp | Fail | Pass |

Mean clean gain: **+0.4654 percentage points**, below the frozen +2 pp
precision requirement for every seed. Mean sigma=.05 absolute gain: **+7.8851
pp**. Mean reduction in degradation: **+7.4197 pp**. All clean and normal-input
No-Fire guards passed. However, under `rgb_zero`, Smoke IoU relative to B0
fell by 5.1605 pp and 15.1968 pp for seeds 204/205; under `thermal_zero`,
No-Fire FP rose for all three seeds. These stress-test effects are not a claim
about all real sensor faults.

S7 is evidence for a **thermal-noise robustness contribution**, not a
demonstrated clean-accuracy architecture improvement. Its UAFM operator is
borrowed/adapted, not established as a novel operator. These are repeated
development-set experiments; the new seeds are not a new independent dataset,
and the utility thresholds are not statistical significance tests.

## Execution and evidence boundary

- The 100-epoch fixed order was B0-203, S7-203, B0-204, S7-204, B0-205,
  S7-205. Formal started/completed/evaluated = **6/6/6**. One S7-204 Windows
  memory error 1455 occurred during validation after epoch 15; the resource
  failure was preserved and the run was recovered from its complete epoch-15
  checkpoint. It is not an effect failure. The final completion audit is PASS.
- No test107 data, statistics, predictions or sample lists were read during
  these experiments. Inputs were verified unchanged. Fire/Heat, Background,
  mIoU and S are record-only, not success gates.
- Local, more complete lightweight evidence bundle:
  `C:\Tmp\FLAME3_B0_S7_100E_FINAL_20260924.zip`, SHA256
  `55A61D0C1D55C1F0A604EE2FE596A8FD24CABAA70CA8CEAAD1876BF49F0A90D1`.
  The final report bytes here match the locally audited report SHA256
  `32A60B0366CF3BD840D79BD8D101AC2326D2DFD6F62818BC7D89EEB610E5712D`.
- [100-epoch source snapshot](code/s7_100e_source_snapshot/) is a lightweight
  copy of the executed adapter/queue/reporting code. It is **not** a runnable
  repository by itself; it depends on the frozen base project, inputs and
  environment, none of which are uploaded as training data or weights here.

## What to decide next

Do not automatically resume the stopped R1 sweep, start S2, retune S8/S7,
train a new structure, or evaluate on test107. For a paper focused on clean
Smoke accuracy, S7 is not the main improvement. A robustness-focused narrative
requires a separate assessment of the modality-zero tradeoffs and independent
confirmation before broad generalization claims. Any new training needs a new
explicit plan and authorization. Preserve the earlier frozen decisions.
