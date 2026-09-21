"""Keep scheduler-enforced idle timing while observing actual external activity."""
from __future__ import annotations

import os
import time

import benchmark_idle as original
from launch_replay_wddm import verify_amendment
from stage2_protocol import write
from wddm_admission import snapshot,blockers

ROOT=original.ROOT
_snapshot=original.gpu_snapshot
_guard=original.install_guard


def observed_snapshot():
    result=_snapshot()
    result['wddm_activity']=snapshot()
    return result


def external(observation):
    return blockers(observation['wddm_activity'],timing=True)


def mark_attempt(allowed):
    write(ROOT/'LATENCY_ATTEMPT.json',dict(pid=os.getpid(),started_unix=time.time(),
          scheduler_stop_on_idle_end=True,automatic_retry=False))
    _guard(allowed)


def main():
    verify_amendment()
    if (ROOT/'LATENCY_REPORT.json').exists() or (ROOT/'LATENCY_FAILURE.json').exists():
        return
    if (ROOT/'LATENCY_ATTEMPT.json').exists():
        write(ROOT/'LATENCY_FAILURE.json',dict(status='PRIOR_ATTEMPT_INTERRUPTED',
              automatic_retry=False,formal_result_eligible=False,
              action='Inspect prior process and shared lock; no automatic lock removal'))
        return
    original.gpu_snapshot=observed_snapshot
    original.external=external
    original.install_guard=mark_attempt
    original.main()


if __name__=='__main__':
    main()
