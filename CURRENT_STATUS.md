# FLAME3 current status

Last verified: 2026-09-21 18:05 China time.

## Completed evidence

- First Stage 2/3 batch: S3, S4, S6, R1, B1, E1, three seeds each, 30 epochs.
- Append batch: S1, R2, R3, three seeds each, 30 epochs, followed by read-only Stage 3.
- Research-arm summary covers S1-S8, R1-R3 and B1. S2, S5, S7 and S8 were not run.
- No evaluated research arm passed the frozen clean-Smoke precision rule.
- R1/R3 thermal-noise behavior remains descriptive, not a formal robustness win,
  because the matching B0/v1.1 epoch26-30 perturbation window was missing.
- E1 is an engineering control: static GMAC proxy reduced 5.066%, with no precision win.

Start with
[`handoffs/structure_20260921/00_READ_ME_FIRST.md`](handoffs/structure_20260921/00_READ_ME_FIRST.md)
and the unified report linked there.

## Authorized baseline replay

A current-environment B0 replay is authorized for seeds 200/201/202, 30 epochs,
retaining epochs26-30 and evaluating clean, thermal noise 0.02/0.05/0.10,
thermal_zero and rgb_zero. It uses the frozen candidate source and input hashes.

At this snapshot:

- Status: `WAITING_GPU`, completed `0/3`; training has not started.
- Explicit blocker: active Sunlogin 3D/video-encode engines.
- Admission: shared project lock plus 60 continuous low-activity seconds.
- The scheduler never closes external applications.
- Historical v1.1 used a different Python/Torch/CUDA/cuDNN environment. Even a
  near-exact metric replay requires provenance review before any formal upgrade.
- Formal B0/E1 latency remains pending and can run only after B0 completion and
  ten minutes of uninterrupted Windows idle time.

Exact state files and the scheduling amendment are under
`handoffs/structure_20260921/07_current_status_snapshot_20260921` and
`handoffs/structure_20260921/06_baseline_replay_authorized_20260921`.

## Preservation

- Verified remote backup: 1458 files, 8,809,315,756 uncompressed bytes.
- Archive: 8,182,829,077 bytes on physical disk E:, separate from source disk D:.
- Archive SHA256: `08459837EE608B5D35AF26C5D434A2B26E66083B950C53E7395370D2F183108D`.
- Backup manifest SHA256: `2843E15D28597C73ADDDD3D795D936301781E2846928E081E14E4B22DF27479E`.
- Off-machine archive transfer is incomplete and therefore not yet verified.

This repository intentionally excludes datasets, images, masks, predictions,
credentials and checkpoint binaries.

## Boundaries

No S2/S5/S7/S8 training, new arm training, 100-epoch extension, automatic four-arm
rerun, threshold change or test107 access is authorized by the current work order.
