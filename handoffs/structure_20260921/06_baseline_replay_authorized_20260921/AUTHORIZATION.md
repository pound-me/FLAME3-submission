# Authorization and frozen replay rules

Date: 2026-09-21. The user approved the preceding three-item plan.

## Scope

1. Preserve 4090 checkpoints and evidence; synchronize code/reports to GitHub.
2. Replay B0/v1.1 seeds 200/201/202, 30 epochs each, keeping every checkpoint in epochs26-30.
3. Evaluate that fixed window under clean, thermal noise 0.02/0.05/0.10, thermal_zero and rgb_zero.
4. Measure B0 and E1 in the same environment only when Windows reports an idle session.

No S2/S5/S7/S8, R4, 100-epoch run, or new candidate training is authorized here. A failed replay does not automatically authorize a four-arm rerun.

## Training identity

Reuse the completed six-arm training loop, source tree, initialization, train493/val134/manual47/NoFire35 inputs, seed order, physical batch8, four workers, AMP, SGD, gradient clip5, learning-rate horizon100, and all augmentation settings. B0 is an exact baseline deepcopy and never receives R1/R3 thermal augmentation. Initialize from the frozen ImageNet file with the original four-channel stem skip, never from a trained checkpoint. The current candidate environment is Python3.11.5/Torch2.1.0+cu121/cuDNN8801. Training retains cudnn TF32=True/matmul TF32=False; the TF32-off numerical-gate supplement does not change training precision.

## Frozen verification before new results

- Sources and input files must match their recorded SHA256 exactly. Effective configs must equal same-seed R1/R2/R3 configs exactly.
- Candidate runtime fields must match Python/Torch/CUDA/cuDNN and training precision exactly.
- Every B0 stage3 clean replay must differ from its own epoch record by strictly less than 1e-6 on every recorded metric.
- Historical recovery test: every same-seed, same-epoch26-30 clean Smoke IoU and No-Fire joint FP ratio must differ from the original v1.1 record by strictly less than 1e-6. This is a conservative near-exact replay check, NOT a newly tuned acceptance band or an effect threshold.
- Record historical environment mismatch separately. The original v1.1 used Python3.9.21/Torch2.6.0+cu118/cuDNN90100, so this replay is explicitly a current-environment B0, not a bitwise recovery of lost checkpoints.
- If historical recovery fails, keep historical conclusions frozen and mark any new cross-run pairing DESCRIPTIVE_ONLY_REPLAY_NOT_RECOVERED. Do not promote robustness, adjust tolerance, or train extra arms automatically. Even if the near-exact metric check passes, provenance differences must remain disclosed and require review before upgrading historical formal claims.
- Original precision/robustness thresholds and all per-seed guards remain unchanged. Positive 2x2 interaction does not override clean guards. Fire/S/mIoU remain record-only.

## Idle latency

Use seed200 epoch30 for B0 and E1, inference-only deployment (no auxiliary heads), 1x4x512x640 synthetic input, AMP FP16, warmup100, ten trials of 200 CUDA-event measurements. Report all 2000 samples, mean/P95/peak allocated and reserved memory, THOP MACs, full environment, and external GPU processes. Same-machine contemporaneous B0 is the timing comparator. Keep historical timing as context only. Never select the fastest rerun. The scheduled task requires ten minutes of Windows idle time and the shared exclusive GPU lock; no desktop application is opened.

## Boundaries

Only the new package, new result directories, and new backups may be written. Old reports, configs, masks, and checkpoint files remain unchanged. No test107 content, no test inference, and no prediction or mask writes. Background processes are never killed to make room. A hash mismatch or nonfinite training output stops the queue for inspection.
