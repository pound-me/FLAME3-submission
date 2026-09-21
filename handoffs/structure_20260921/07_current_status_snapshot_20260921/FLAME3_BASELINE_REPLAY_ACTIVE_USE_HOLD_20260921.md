# FLAME3 baseline replay: active-use hold

Recorded 2026-09-21 17:41 China time.

The user reported that someone is using the remote computer and asked whether
the work would affect them. To avoid an automatic training start, the assistant
held its own unstarted queue and transfer. No other person's application was stopped.

## Confirmed remote state

- Baseline B0 training has NOT started: seeds completed 0/3, no training child.
- Original queue PID 43788 was WAITING_GPU; its waiting task is now stopped.
- Tasks FLAME3_B0_Replay_20260921 and FLAME3_B0_E1_Idle_Latency_20260921 are disabled.
- Remote hold record: flame3_baseline_replay_20260921_v1/ACTIVE_USE_HOLD_20260921.json.
- The old QUEUE_STATUS.json is retained as prior status; the hold record supersedes it.
- No test107 access, no desktop application launched, no other arms started.

## Preservation

- Verified remote backup is on E: (physical disk1), separate from source D: (disk0).
- Directory: E:/FLAME3_BACKUPS/structure_20260921_v1.
- 1458 files, including 135 retained candidate epoch26-30 checkpoints.
- ZIP bytes: 8182829077.
- ZIP SHA256: 08459837EE608B5D35AF26C5D434A2B26E66083B950C53E7395370D2F183108D.
- MANIFEST SHA256: 2843E15D28597C73ADDDD3D795D936301781E2846928E081E14E4B22DF27479E.
- Off-machine transfer is paused and NOT verified; partial ZIP is 43876352 bytes.
- The partial archive and two downloaded manifests are preserved locally under
  backups/flame3_4090_preservation_20260921. Do not format the backup disk.

## Code and synchronization

The original frozen 38-file baseline replay package remains installed remotely.
Its MANIFEST SHA256 is 2395FB0475A2F0377599C4DD804D57E969A0636A291B548D4CE40760E8082535.
Seven original tests and remote preflight passed before the hold.

Six new WDDM scheduling amendment files plus a separate manifest exist locally only under
staging/flame3_baseline_replay_20260921_v1. Seven admission tests pass locally,
but these wrappers have NOT been installed, remotely validated, or frozen in
SCHEDULER_AMENDMENT_MANIFEST.json. Do not launch them as if deployment were complete.

GitHub branch codex/structure-results-20260921 is prepared locally in
github_sync_flame3_20260824, with uncommitted handoffs/structure_20260921 files.
No new commit or push has been made. GitHub synchronization remains pending.

## Resume only after confirming availability

1. Reconfirm machine availability and current GPU processes; never stop others.
2. Inspect the remote hold and preserved original queue status before a restart.
3. Finish validation/freezing of scheduling-only additions if using them. Preserve
   the original training/evaluation workers, sources, manifests and result rules.
4. Resume or separately restart the partial backup with full SHA256 verification.
5. Start only the authorized B0 seeds200/201/202, 30 epochs, followed by the fixed
   six-condition epoch26-30 evaluation. Historical runtime mismatch remains disclosed.
6. Run formal B0/E1 latency only in an uninterrupted idle period, serial with training.
7. Finish GitHub evidence synchronization and record the actual commit.

No new candidates, second-batch arms, 100-epoch runs, or automatic four-arm reruns
are authorized by the earlier approval.

## Resume update, 2026-09-21 18:02 China time

The user later authorized continuation. WDDM scheduling amendment v2 was installed
and verified remotely after 15 contract tests passed. Its manifest SHA256 is
`1750E8710CF165B9BA121F286EA052B1F69E14254B02553EE37DA909A2E0E0B1`.
The old waiting status was archived, and the B0 queue plus idle latency task were
enabled. This supersedes the disabled-task state above, while preserving it as
historical evidence of the hold.

At the captured status snapshot, B0 remained `WAITING_GPU`, completed `0/3`, with
no training child. Active Sunlogin 3D/video-encode engines were the explicit blocker.
The queue requires 60 continuous seconds of low activity before seed200 can start.
It does not close or suspend any external application. The formal latency task
remains gated on B0 completion and ten minutes of Windows idle time.
