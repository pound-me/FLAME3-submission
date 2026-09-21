# FLAME3 structure screen stage 3 authorization

This package implements the supplement `工单补充3_数值等价门精度模式修订_20260919.md` with SHA256 `A2F962BC63875205B500FFC77F3D38D6BA13F72D702099B3817D28944C7EDC08`.

The supplement is dated 2026-09-19, while the actual execution date is 2026-09-18. The `20260919` suffix is retained as the work-order identifier, not as an execution timestamp.

- Evaluate the completed S3, S4, S6, R1, B1 and E1 runs for seeds 200, 201 and 202.
- Report clean Smoke, robustness and efficiency axes. S, mIoU and Fire-side values are record-only.
- Reuse `evaluate_test107_posthoc.py::perturb`; frozen source SHA256 is `BE3761AFCB7EECDCEBDB7172CEA3BCDD664030DA5C57F6E94D95EC8E75911ACC`.
- Read only val134 and the frozen A001-A047 manual targets. Do not access test107 in any form.
- Do not train, backpropagate, save predictions or modify any checkpoint, label, configuration or original report.
- Stage 3 owns the shared GPU lock. S1/R2/R3 revised gates and append training must wait until stage 3 releases it.
- The v1.1 runs retained epoch100 `last.pth` and later best checkpoints, but not epoch26-30 checkpoints. Clean comparison uses the frozen epoch26-30 metric window. Robust candidate windows are fully reported; v1.1 epoch100 is context-only, while the exact positive robustness decision remains not determinable rather than fabricating a baseline window.
