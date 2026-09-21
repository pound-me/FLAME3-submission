"""Scheduling-only amendment; frozen training/evaluation workers are unchanged."""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

from stage2_protocol import read,write,sha,verify_package,SEEDS
from launch_replay import run_child
from wddm_admission import snapshot,blockers

ROOT=Path(__file__).resolve().parent
LOCK=ROOT.parent/'flame3_structure_screen_20260917_gpu.lock'


def verify_amendment():
    verify_package(ROOT)
    for name,expected in read(ROOT/'SCHEDULER_AMENDMENT_MANIFEST.json')['files'].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'Scheduling amendment hash mismatch: {name}')


def wait_gpu(state, *, owns_lock=False):
    stable_since=None
    while True:
        observed=snapshot()
        reasons=blockers(observed)
        if LOCK.exists() and not (owns_lock and LOCK.read_text(encoding='utf-8')==str(os.getpid())):
            reasons.append('shared GPU lock owned elsewhere')
        if reasons:
            stable_since=None
        elif stable_since is None:
            stable_since=time.monotonic()
        seconds=0 if stable_since is None else time.monotonic()-stable_since
        state.update(status='WAITING_GPU',updated_unix=time.time(),admission_stable_seconds=seconds)
        write(ROOT/'QUEUE_STATUS.json',state)
        write(ROOT/'GPU_ADMISSION_STATUS.json',dict(observation=observed,blockers=reasons,
              continuous_low_activity_seconds=seconds,required_seconds=60,observed_unix=time.time()))
        if not reasons and seconds>=60:
            return
        time.sleep(5)


def main():
    verify_amendment()
    if (ROOT/'QUEUE_STATUS.json').exists():
        raise RuntimeError('Existing queue status, inspect before any restart')
    state=dict(status='WAITING_GPU',pid=os.getpid(),seeds=list(SEEDS),completed=[],
               started_unix=time.time(),test107_read=False,other_arm_training=False,
               scheduling_amendment_sha256=sha(ROOT/'SCHEDULER_AMENDMENT_MANIFEST.json'))
    if read(ROOT/'BACKUP_STATUS.json')['status']!='REMOTE_BACKUP_VERIFIED':
        raise RuntimeError('Verified remote backup required')
    wait_gpu(state)
    with LOCK.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for index,seed in enumerate(SEEDS):
            if index:
                wait_gpu(state,owns_lock=True)
            state.update(status='RUNNING_B0',current_seed=seed,updated_unix=time.time())
            write(ROOT/'QUEUE_STATUS.json',state)
            run_child('run_baseline.py',['--arm','B0','--seed',str(seed)],f'B0_seed{seed}.log')
            if read(ROOT/'runs/B0'/f'seed{seed}'/'RESULT.json')['status']!='COMPLETE_30_EPOCHS':
                raise RuntimeError('Incomplete baseline training')
            state['completed'].append(seed)
            write(ROOT/'QUEUE_STATUS.json',state)
        wait_gpu(state,owns_lock=True)
        state.update(status='EVALUATING_15_CHECKPOINTS',current_seed=None)
        write(ROOT/'QUEUE_STATUS.json',state)
        run_child('evaluate_baseline.py',[],'baseline_window_evaluation.log')
        state.update(status='COMPLETE_B0_TRAINING_AND_EVALUATION',finished_unix=time.time(),
                     latency_status='WAITING_WINDOWS_IDLE_TASK')
        write(ROOT/'QUEUE_STATUS.json',state)
    except BaseException as exc:
        state.update(status='STOPPED_FOR_AUDIT',error=repr(exc),traceback=traceback.format_exc())
        write(ROOT/'QUEUE_STATUS.json',state)
        raise
    finally:
        if LOCK.exists() and LOCK.read_text(encoding='utf-8')==str(os.getpid()):
            LOCK.unlink()


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        write(ROOT/'QUEUE_WDDM_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
