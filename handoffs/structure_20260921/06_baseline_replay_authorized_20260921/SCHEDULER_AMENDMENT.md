# Scheduling-only amendment, 2026-09-21

The first launcher remained WAITING_GPU and never started a training worker.
On this Windows WDDM host, the compute-app query included desktop contexts.
Read-only observation found GPU utilization 1%, approximately 2000 MiB allocated,
no Python CUDA worker, and only approximately 0.08% desktop engine activity.
No application is stopped, and no model/data/config/precision/effect threshold changes.

The original 38-file package and manifest remain unchanged. Additional wrappers
have a separate SHA256 manifest and reuse the original training/evaluation workers.
The initial waiting status is archived before replacing our own waiting task.

The first remote runtime sample found that `nvidia-smi` emitted a process path in
the Windows legacy code page while Python was forced to UTF-8. No task was changed
and no training started. Amendment v2 queries only ASCII PIDs from `nvidia-smi`,
then resolves Unicode process names through Base64-encoded PowerShell JSON.

## Training admission

Use the shared project GPU lock and at least 60 continuous seconds of low activity.
Sample three Windows GPU Engine counters, spaced two seconds apart, on each check.
Block other Python GPU contexts, total utilization above 5%, allocated memory above
4096 MiB, non-desktop/unknown engine activity above 0.05%, or desktop activity above
1%. MATLAB/IDL themselves are NOT desktop-whitelisted. Idle contexts may remain.
Check again between seeds and before evaluation. This does not reserve the GPU
against unrelated jobs started in the future; all project queues share the lock.

## Latency

Keep the Windows scheduled task's ten-minute idle requirement and StopOnIdleEnd.
During measurement observe external engines, with a stricter desktop activity
limit of 0.5% and unchanged non-desktop 0.05%. The current timing process is excluded.
Only the runtime occupancy filter changes; input, warmup, repetitions, precision,
checkpoint selection and aggregation are unchanged. All observations are saved.

If Windows terminates the task when activity resumes, a prior-attempt marker
prevents selecting a faster retry. The next task run records an interrupted result
and does not automatically remove the stale lock. That case requires inspection.
No formal timing result is claimed before a complete uninterrupted idle run.
