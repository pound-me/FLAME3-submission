# FLAME3 recent-results handoff

Package date: 2026-09-21

Live status entry: [`../../CURRENT_STATUS.md`](../../CURRENT_STATUS.md).

Latest addition: [`08_b0_completed_and_closeout_20260921/README.md`](08_b0_completed_and_closeout_20260921/README.md).
Read it first: B0 replay and evaluation are now complete, historical recovery
failed, and the formal-latency attempt failed its occupancy monitoring. The
18:05 waiting snapshot retained below is historical, not the current state.

Covered work: 2026-09-17 through 2026-09-19 network-structure screening, engineering gates, Stage 2 training records, Stage 3 read-only evaluation, and the final twelve-arm/three-axis summary.

## Read in this order

1. `02_reports/FLAME3_STRUCTURE_12_ARM_THREE_AXIS_REPORT_20260919.md`
2. `02_reports/FLAME3_STRUCTURE_STAGE3_AND_REVISED_GATE_REPORT_20260919.md`
3. `02_reports/FLAME3_STRUCTURE_STAGE1_ENGINEERING_REPORT_20260918.md`
4. `01_work_orders/预注册_网络结构改进筛选_修订版v2_20260917.md`
5. `01_work_orders/工单补充3_数值等价门精度模式修订_20260919.md`

Use `03_evidence` when checking exact numerical values and hashes. Use `04_arm_designs` and `05_code` when reviewing architecture definitions and implementation details.

## Current frozen status

- The twelve research arms are S1-S8, R1-R3, and B1. E1 is a separate engineering control.
- Completed three-seed, 30-epoch evaluation: S1, S3, S4, S6, R1, R2, R3, and B1.
- S2, S5, S7, and S8 were not authorized for the second batch and were not run. Missing results must not be imputed.
- No evaluated research arm passed the frozen clean-Smoke precision rule.
- R1 and R3 showed comparatively small descriptive Smoke drops under thermal noise, but the formal robustness result is `NOT_DETERMINABLE_BASELINE_WINDOW_MISSING` because matching v1.1 epoch26-30 perturbation checkpoints do not exist.
- The R1/R2/R3/v1.1 2x2 table is descriptive only: the retained B0 perturbation reference uses v1.1 epoch100 while R1/R2/R3 use epoch26-30.
- E1 reduced the static GMAC proxy by 5.066% and is retained only as an engineering-efficiency control, not as a precision win.
- Fire/Heat, selection score S, and three-class mIoU are record-only and do not trigger a positive decision.
- Formal latency is still deferred.
- A B0/v1.1 current-environment replay has now been authorized to reconstruct the
  missing epoch26-30 paired window. At the 2026-09-21 18:05 snapshot it is
  `WAITING_GPU`, completed `0/3`, blocked by an active Sunlogin remote session.
- Do not treat authorization or a running scheduler as a completed replay.
- Historical and current candidate software environments differ; the frozen
  provenance-review rule remains in force even if numerical recovery passes.

## Safety and provenance

- Stage 3 performed no training and did not write prediction files.
- The Stage 3 audits report `test107_read: false`.
- Input and context hashes remained unchanged.
- This package contains no dataset images, masks, model weights, predictions, credentials, or bulky run logs.
- Some evidence files retain original Windows paths for provenance. Those paths are historical references and are not expected to exist on another machine.

## Package layout

- `01_work_orders`: frozen instructions and later precision-mode supplement.
- `02_reports`: chronological reports and the final unified table/report.
- `03_evidence`: machine-readable Stage 3 summaries, tables, attribution data, status, and preflight evidence.
- `04_arm_designs`: design and static-cost record for every research arm plus E1.
- `05_code`: architecture implementation, engineering checks, Stage 2/3 runners, tests, and unified-report builder.
- `06_baseline_replay_authorized_20260921`: frozen B0 replay package and WDDM scheduling amendment.
- `07_current_status_snapshot_20260921`: exact remote queue, backup and scheduler status files.
- `08_b0_completed_and_closeout_20260921`: completed B0 report/results, final v4 scheduler, latency failure audit, and backup resumption evidence.
- `FILE_SHA256.csv`: original 2026-09-19 result-package integrity manifest.
- `FILE_SHA256_20260921.csv`: refreshed manifest covering the complete current handoff.

## Recommended next decision

The B0 replay is complete but did not recover historical metrics. Keep the new
pairing descriptive and review provenance before deciding on any new experiment.
Do not describe any tested arm as a confirmed accuracy or robustness improvement,
and do not start S2/S5/S7/S8 or redesign training automatically.
