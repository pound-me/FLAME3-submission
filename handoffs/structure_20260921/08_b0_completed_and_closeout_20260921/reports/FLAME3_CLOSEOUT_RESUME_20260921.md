# FLAME3 closeout resume, 2026-09-21

Scope: continue the previously paused preservation, final evidence synchronization
and B0/E1 strict-idle timing work. No tuning or additional training is authorized.

## Completed result

B0 seeds 200/201/202 completed 30 epochs and all 15 retained checkpoint units
completed the six-condition evaluation. The narrative report remains unchanged:
`FLAME3_BASELINE_REPLAY_AND_WINDOW_EVALUATION_20260921.md`.
SHA256: `7CEDCC9F403B2EA06F9A4DCB25C1A773E2DD0A8E73FF7A83839A218074875D1A`.

Historical recovery passed 0/15 units. The new pairing remains
`DESCRIPTIVE_ONLY_REPLAY_NOT_RECOVERED`; formal upgrading is not allowed.

## Formal latency: monitoring failure, not a performance result

The task was re-enabled at 20:38:20 with the original ten-minute Windows idle
requirement, StopOnIdleEnd and no automatic retry after a measured attempt.
The attempt marker records PID75900 and started_unix 1789994742.9279795.
During a per-trial occupancy check, PowerShell Get-Counter returned exit code 1.
The preserved traceback points to `benchmark_idle.py:99`,
`benchmark_idle_wddm.py:19`, and `wddm_admission.py:34`.

No LATENCY_REPORT.json was produced. On 20:55:49 inspection the lock was absent,
the task was disabled, and `formal_result_eligible` and `automatic_retry` were
both false. The original counter stderr is not in the recorded exception, so
the underlying Windows counter error is not yet established. It is not evidence
that the model is slower or faster. No measured retry or counter repair was run.

Failure SHA256: `50D94209610AEC1CBF5271E2E102BC89FBD3792970F87E0C4085E694B9E22021`.
Attempt SHA256: `6A03B494782EA0EA3CA0A253AE671AA5FFDBE61089378539B66B0E47CED1E7BF`.

## Backups

The verified 8,182,829,077-byte remote archive remains on
`E:\FLAME3_BACKUPS\structure_20260921_v1`.
Archive SHA256: `08459837EE608B5D35AF26C5D434A2B26E66083B950C53E7395370D2F183108D`.

Off-machine transfer resumed at 20:41. The 43,876,352-byte original partial plus
62,087,168-byte remainder were verified as the exact remote archive prefix before
appending. The transfer keeps a 4 MiB/s cap, but observed throughput is much lower.
Tailscale reported DERP(nue), a 4.557-second ping, and a subsequent timeout.
No proxy, tunnel, firewall or external application setting was modified.

At 20:50 the combined fragments held 124,174,336 bytes (1.517%). This is a progress
snapshot, not a verified full backup or a completion-time promise.

At 20:58:31 the transport recorded Connection reset and SSH exit code 255.
At 21:00 the receiver was no longer running; all 130,859,008 bytes (1.5992%)
remained on disk. Later minimal SSH probes also timed out. The transfer is
interrupted, not actively running, and no full off-machine verification exists.

The original archive predates B0 and contains none of its 15 new checkpoints.
`scripts/preserve_flame3_b0_increment_20260921.ps1` was prepared to preserve these
and the frozen source/results in a new E-drive directory, with source, copied-file
and archive-entry SHA256 checks. Remote script transfer/start is unconfirmed
after timeouts. The increment must not be represented as complete.

Do not format D:, E: or the remote machine until all required weights have a
verified independent copy. GitHub contains source and evidence, not model weights.

## Verification

- Original remote package: 38/38 file SHA256 matched.
- Final scheduler v4: 6/6 file SHA256 matched.
- Local replay/admission contract tests: 19/19 passed.
- Updated backup receiver: PowerShell parse plus empty/single-file SHA256 tests passed.
- Incremental helper: PowerShell parse passed; remote execution not verified.
- New result transfers matched their remote SHA256.
- No test107 reads, extra training, threshold changes or unrelated process stops.

GitHub handoff: `handoffs/structure_20260921/08_b0_completed_and_closeout_20260921`.
Its CURRENT_STATUS.md separates completed experiments from failed or pending
operational work. Historical reports are retained without edits.
