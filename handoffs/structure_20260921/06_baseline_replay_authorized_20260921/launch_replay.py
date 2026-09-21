"""Hidden sequential B0 queue with exclusive GPU ownership."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from stage2_protocol import read, write, verify_package, SEEDS

ROOT = Path(__file__).resolve().parent
LOCK = ROOT.parent / "flame3_structure_screen_20260917_gpu.lock"


def external_compute():
    result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    return [s.strip() for s in result.stdout.splitlines() if s.strip().isdigit()]


def run_child(script, args, log_name):
    log = ROOT/'logs'/log_name
    log.parent.mkdir(exist_ok=True)
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable,'-B','-X','utf8','-u',str(ROOT/script),*args],
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f'{script} failed: {result.returncode}, see {log}')


def main():
    verify_package(ROOT)
    status = ROOT/'QUEUE_STATUS.json'
    if status.exists():
        raise FileExistsError('Existing queue status: inspect before any restart')
    state = dict(status='WAITING_BACKUP', pid=os.getpid(), seeds=list(SEEDS), completed=[],
                 started_unix=time.time(), test107_read=False, other_arm_training=False)
    write(status,state)
    while True:
        if (ROOT/'BACKUP_FAILURE.json').exists():
            raise RuntimeError('Backup failed, training blocked')
        b = ROOT/'BACKUP_STATUS.json'
        if b.exists() and read(b)['status']=='REMOTE_BACKUP_VERIFIED':
            break
        time.sleep(15)
    # The lock-file convention is shared with all preceding queues.
    while LOCK.exists() or external_compute():
        state.update(status='WAITING_GPU',updated_unix=time.time())
        write(status,state)
        time.sleep(30)
    with LOCK.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for seed in SEEDS:
            if external_compute():
                raise RuntimeError('Another compute process appeared; no processes will be stopped')
            state.update(status='RUNNING_B0',current_seed=seed,updated_unix=time.time())
            write(status,state)
            run_child('run_baseline.py',['--arm','B0','--seed',str(seed)],f'B0_seed{seed}.log')
            record=read(ROOT/'runs/B0'/f'seed{seed}'/'RESULT.json')
            if record['status']!='COMPLETE_30_EPOCHS':
                raise RuntimeError('B0 did not complete 30 epochs')
            state['completed'].append(seed)
            write(status,state)
        state.update(status='EVALUATING_15_CHECKPOINTS',current_seed=None)
        write(status,state)
        run_child('evaluate_baseline.py',[],'baseline_window_evaluation.log')
        state.update(status='COMPLETE_B0_TRAINING_AND_EVALUATION',finished_unix=time.time(),
                     latency_status='WAITING_WINDOWS_IDLE_TASK')
        write(status,state)
    except BaseException as exc:
        state.update(status='STOPPED_FOR_AUDIT',error=repr(exc),traceback=traceback.format_exc())
        write(status,state)
        raise
    finally:
        if LOCK.exists() and LOCK.read_text(encoding='utf-8')==str(os.getpid()):
            LOCK.unlink()


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        write(ROOT/'QUEUE_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
