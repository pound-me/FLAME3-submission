# B0 completion and closeout, 2026-09-21

This additive handoff supersedes the operational waiting snapshot in folder 07.
Frozen experiment reports and the original source package in folder 06 are not
rewritten. The later scheduling-only v4 files are supplied separately here.

## Main result

Read [the completed B0 report](reports/FLAME3_BASELINE_REPLAY_AND_WINDOW_EVALUATION_20260921.md).
All three 30-epoch seeds and 15 fixed-window checkpoint evaluations are complete.
Historical recovery passed 0/15 units; all new pairing is descriptive only and
no historical formal claim is upgraded. No new training was started on resume.

The report is a frozen completion snapshot. Its final operational paragraph
predates the resumed latency attempt; use the status below for later events.

## Later events

- 20:38: the strict ten-minute Windows-idle latency task was re-enabled.
- 20:41: large-archive transfer resumed after the existing 105,963,520-byte
  prefix, including both preserved fragments, matched the remote SHA256.
- 20:43: 38 original source and six v4 scheduler file hashes matched.
- About 20:45: latency attempt began. GPU-engine occupancy counter collection
  failed during a per-trial check. No formal latency report was produced.
- 20:55: the failed latency task was disabled. Failure and attempt evidence
  remain unchanged; no automatic timing retry was performed.
- The original large backup excludes the newly generated B0 checkpoints.
  An explicit-allowlist incremental helper was prepared, but its remote start
  was not confirmed after connection timeouts. Do not treat it as completed.
- 20:58: the large-archive SSH stream reset and exited with code 255. All
  130,859,008 received bytes are preserved (1.5992%); the receiver is stopped.
  Subsequent connectivity probes timed out. Neither backup is claimed complete.

## Contents

- `reports`: unmodified completed B0 narrative report.
- `results`: exact remote B0 results, queue/evaluation states, and source integrity
  snapshot. The latency state here is an earlier waiting snapshot.
- `scheduler_v4`: final scheduling-only amendment with its own manifest. These
  files overlay folder 06 only when reproducing that already completed package.
- `closeout_status`: resume, prefix verification and subsequent failure audit.
- `backup_helpers`: transport and incremental preservation code; code presence
  is not proof that a backup finished.

The authoritative latency state is `closeout_status/latency_failure/`, not the
older `results/LATENCY_STATUS.json`. Full off-machine backup verification remains
pending. No datasets, masks, predictions or checkpoint binaries are in this folder.

## Next decisions

Keep the original historical recovery and arm pass thresholds unchanged. Repair
and review the occupancy-monitor failure before explicitly authorizing another
timing attempt. Complete the separate B0 checkpoint preservation and the large
off-machine transfer before considering any formatting of the 4090 machine.
Hyperparameter tuning has been discussed only; it has not been authorized or run.
