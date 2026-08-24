# FLAME3 night-run results (2026-08-24)

## Outcome

| Job | Status | Key result |
|---|---|---|
| 0 | Complete; queue allowed | test107 manual-machine aggregate Fire IoU `0.234093`; best translation `dy=+2, dx=0` added only `0.006533` IoU. |
| 1 | Complete | val47 Fire/Heat delta `+0.013151` (95% CI `[+0.004137, +0.022787]`); test107 delta `-0.006879` (95% CI `[-0.010903, -0.002760]`). |
| 2 | Complete with one missing seed | Fusion/RGB-only/Thermal-only mean S: `0.722277` / `0.383231` / `0.638501`; Thermal-only seed 201 reproduced a non-finite-gradient failure. |
| 3 | Stopped at self-check | Ignore pixels changed full training loss: `2.2008969784 != 2.2227547169`. |
| 4 | Not started | Prohibited by the Job 3 hard gate; zero training and inference runs. |
| 5 | Complete | Frozen CSV tables and SVG figures generated without training or inference. |

## Files

- [Morning summary](../../../docs/NIGHT_RUN_SUMMARY_20260824.md)
- [test107 label-gap report](../../../docs/FLAME3_LABEL_GAP_TEST107_BASIS_20260824.md)
- [Job 1 statistical report](../../../docs/FLAME3_NIGHT_JOB1_STATISTICS_20260824.md)
- [Job 1 statistics](job1_statistics/)
- [Job 2 modality ablation](job2_input_ablation/)
- [Job 3 failure evidence](job3_engineering_gate/)
- [Job 4 skip audit](job4_skip_audit/)
- [Paper tables and figures](paper_assets/)
- [Label-gap tables and aggregate plots](../../../experiments/flame3_label_gap_test107_basis_20260824/)
- [Submission errata](../label_gap/)

## Public-package boundary

This public branch intentionally excludes test107 representative-case images, raw images, masks, predictions, checkpoints, and manifests containing local absolute paths. The complete read-only audit remains preserved in the local evidence archive. No test107 inference was performed for Jobs 0, 1, or 5.
