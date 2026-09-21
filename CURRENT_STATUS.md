# FLAME3 current status

Updated: 2026-09-21, China time. Final evidence below supersedes the earlier
18:05 waiting snapshot. Historical reports remain unchanged.

## Read first

1. [B0 completion and closeout](handoffs/structure_20260921/08_b0_completed_and_closeout_20260921/README.md)
2. [B0 replay report](handoffs/structure_20260921/08_b0_completed_and_closeout_20260921/reports/FLAME3_BASELINE_REPLAY_AND_WINDOW_EVALUATION_20260921.md)
3. [Original twelve-arm report](handoffs/structure_20260921/02_reports/FLAME3_STRUCTURE_12_ARM_THREE_AXIS_REPORT_20260919.md)

## Completed experiments

- First batch and append: S1, S3, S4, S6, R1, R2, R3, B1 and engineering control
  E1, seeds 200/201/202, 30 epochs, followed by Stage 3 evaluation.
- S2/S5/S7/S8 remain not run. No new tuning or candidate training was started.
- B0 replay is COMPLETE for all three seeds. Its 15 retained epoch26-30
  checkpoints were evaluated under all six frozen conditions.
- Historical recovery passed 0/15 checkpoint units. Software environments differ.
- Pairing status: `DESCRIPTIVE_ONLY_REPLAY_NOT_RECOVERED`.
- `formal_upgrading_allowed: false`. Historical conclusions are not upgraded.
- R1/R3 show descriptive thermal-noise robustness signals, but no evaluated
  research arm passes the frozen complete rule. Fire/Heat, S and mIoU remain
  record-only. E1 remains an engineering control, not a research innovation win.

See the replay report for the descriptive per-arm table. A failed historical
recovery test does not prove that every possible setting of an arm is ineffective.
It also does not authorize tuning, extra training or relaxed thresholds.

## Formal latency

The B0/E1 idle task was resumed at 20:38. It began an attempt at about 20:45,
but Windows GPU-engine counter collection failed during a per-trial occupancy
check. No complete formal latency report was produced.

- Failure evidence and attempt marker are retained in the closeout package.
- The task was disabled again at 20:55; automatic retry is false.
- The project lock was released. No timing result is eligible for reporting.
- Repair/protocol review and explicit reauthorization are needed before another
  measured attempt. No external application or unrelated process was stopped.

## Preservation

- Existing verified remote backup: 1458 files, including 135 candidate-window
  checkpoints, on physical disk E:, separate from source disk D:.
- Archive size: 8,182,829,077 bytes.
- SHA256: `08459837EE608B5D35AF26C5D434A2B26E66083B950C53E7395370D2F183108D`.
- Off-machine transfer resumed at 20:41. The existing 105,963,520-byte prefix
  matched the remote prefix SHA256. Full off-machine verification is pending.
- At 20:58 the SSH connection reset and the receiver exited with code 255.
  The preserved fragments total 130,859,008 bytes (1.5992%); no receiver remains
  running. Later SSH probes timed out, so resumption awaits remote connectivity.
- The original archive predates the new B0 replay and DOES NOT contain its 15
  new window checkpoints. A separate B0 incremental backup helper is prepared;
  deployment/start was not confirmed after repeated remote connection timeouts.
- Do not format the source or the backup disk on the strength of this snapshot.

The repository contains reports, code, status and hashes, not checkpoint binaries,
dataset images, masks, predictions or credentials. GitHub synchronization is not
a replacement for the complete weight backup.

## Integrity and boundaries

The 20:43 remote snapshot verified all 38 original package files and all six v4
scheduling-amendment files. Nineteen local replay/admission contract tests passed.
The copied B0 result artifacts matched remote SHA256 values.

No test107 content was read. No extra training, threshold changes, new candidate
arms, 100-epoch extension or automatic four-arm rerun was started or authorized
by this closeout resume. The tuning discussion remains a proposal only.
