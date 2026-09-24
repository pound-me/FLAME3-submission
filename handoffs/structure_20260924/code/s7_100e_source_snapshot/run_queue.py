from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
from audit_runtime import BUNDLE, PROTO, ROOT, WORK_SHA, package_identity, read, sha, write
from common import ARMS, SEEDS, run_path
from reporting import current, generate

GLOBAL_LOCK = BUNDLE / 'flame3_b0_s7_100e_20260924_gpu.lock'
OTHER_LOCKS = (
    BUNDLE / 'flame3_s8_grid_followup_20260923_gpu.lock',
    BUNDLE / 'flame3_structure_screen_20260917_gpu.lock',
    BUNDLE / 'experiments/flame3_4060_structure_20260922_v1/GPU.lock',
)
SCHEDULE = [(arm, seed) for seed in SEEDS for arm in ARMS]


def child(script, args, label):
    path = ROOT / 'logs' / f'{label}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if path.exists() else 'x'
    with path.open(mode, encoding='utf-8') as stream:
        if mode == 'a':
            stream.write('\n=== RESUME INVOCATION ' + time.strftime('%Y-%m-%d %H:%M:%S') + ' ===\n')
        stream.flush()
        return subprocess.run(
            [sys.executable, '-B', '-u', str(ROOT / 'source_snapshot' / script), *args],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).returncode


def update(queue, status, phase=None, arm=None, seed=None):
    queue.update(
        status=status,
        current=dict(phase=phase, arm=arm, seed=seed) if phase else None,
        updated_unix=time.time(),
        pid=os.getpid(),
    )
    write(ROOT / 'QUEUE_STATUS.json', queue)
    current(queue)


def process_alive(pid):
    command = f"if(Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue){{'YES'}}else{{'NO'}}"
    result = subprocess.check_output(
        ['powershell.exe', '-NoProfile', '-Command', command],
        text=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).strip()
    return result == 'YES'


def start_resource_preflight():
    command = "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    text = subprocess.check_output(
        ['powershell.exe', '-NoProfile', '-Command', command],
        text=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).strip()
    rows = json.loads(text) if text else []
    if isinstance(rows, dict):
        rows = [rows]
    others = [row for row in rows if int(row['ProcessId']) != os.getpid()]
    if others:
        raise RuntimeError('OTHER_PYTHON_PROCESS_PRESENT_NO_GPU_COMPETITION: ' + json.dumps(others, ensure_ascii=False))

    gpu_text = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).strip()
    parts = [part.strip() for part in gpu_text.split(',')]
    if len(parts) != 4 or parts[0] != 'NVIDIA GeForce RTX 4060 Ti':
        raise RuntimeError('AUTHORIZED_GPU_RESOURCE_QUERY_MISMATCH: ' + gpu_text)

    memory_command = "$os=Get-CimInstance Win32_OperatingSystem; [pscustomobject]@{FreePhysicalMemory=$os.FreePhysicalMemory;FreeVirtualMemory=$os.FreeVirtualMemory}|ConvertTo-Json -Compress"
    memory = json.loads(subprocess.check_output(
        ['powershell.exe', '-NoProfile', '-Command', memory_command],
        text=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).strip())
    snapshot = dict(
        checked_unix=time.time(),
        other_python_processes=[],
        gpu_name=parts[0],
        gpu_total_mib=int(parts[1]),
        gpu_free_mib=int(parts[2]),
        gpu_utilization_percent=int(parts[3]),
        free_physical_gib=float(memory['FreePhysicalMemory']) / 1024 ** 2,
        free_virtual_gib=float(memory['FreeVirtualMemory']) / 1024 ** 2,
        thresholds=dict(gpu_free_mib=12000, free_physical_gib=6.0, free_virtual_gib=12.0),
        unrelated_processes_terminated=False,
    )
    if snapshot['gpu_free_mib'] < 12000 or snapshot['free_physical_gib'] < 6.0 or snapshot['free_virtual_gib'] < 12.0:
        raise RuntimeError('INSUFFICIENT_START_RESOURCES_NO_AUTOMATIC_CLEANUP: ' + json.dumps(snapshot, ensure_ascii=False))
    return snapshot


def acquire_lock(resume):
    recovery = None
    if GLOBAL_LOCK.exists():
        raw = GLOBAL_LOCK.read_bytes()
        try:
            owner = json.loads(raw.decode('utf-8'))
        except Exception:
            owner = {'pid': None, 'raw_sha256': sha(GLOBAL_LOCK)}
        if owner.get('pid') and process_alive(owner['pid']):
            raise RuntimeError('ACTIVE_GPU_LOCK_OWNER_PRESENT')
        if not resume:
            raise RuntimeError('STALE_GPU_LOCK_REQUIRES_RESUME_AUDIT')
        target = ROOT / 'audit/recovery' / f'STALE_GPU_LOCK_{time.time_ns()}.json'
        recovery = dict(
            status='ARCHIVED_STALE_TASK_LOCK',
            source=str(GLOBAL_LOCK),
            source_sha256=sha(GLOBAL_LOCK),
            source_bytes=len(raw),
            owner=owner,
            archived_unix=time.time(),
        )
        write(target, recovery)
        GLOBAL_LOCK.unlink()
    if any(path.exists() for path in OTHER_LOCKS):
        raise RuntimeError('OTHER_FLAME3_GPU_LOCK_PRESENT')
    descriptor = os.open(GLOBAL_LOCK, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    try:
        os.write(descriptor, json.dumps(dict(pid=os.getpid(), protocol=PROTO, started_unix=time.time())).encode('utf-8'))
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        if GLOBAL_LOCK.exists():
            GLOBAL_LOCK.unlink()
        raise
    return descriptor, recovery


def archive_damaged_access_log(path, valid_rows, error):
    target = ROOT / 'audit/recovery/damaged_access_logs' / f'{path.stem}_{time.time_ns()}{path.suffix}'
    target.parent.mkdir(parents=True, exist_ok=True)
    record = dict(
        source=str(path),
        source_sha256=sha(path),
        bytes=path.stat().st_size,
        valid_prefix_rows=len(valid_rows),
        invalid_tail=repr(error),
        denied_in_valid_prefix=sum(row.get('permitted') is False for row in valid_rows),
        test107_hits_in_valid_prefix=sum('test107' in str(row.get('path', '')).lower() for row in valid_rows),
        target=str(target),
    )
    shutil.move(str(path), str(target))
    write(target.with_suffix('.recovery.json'), record)
    if record['denied_in_valid_prefix'] or record['test107_hits_in_valid_prefix']:
        raise RuntimeError('DAMAGED_ACCESS_LOG_VALID_PREFIX_VIOLATION')
    return record


def archive_stale_atomic_temps():
    candidates = []
    seen = set()
    for pattern in ('*.tmp', '*.tmp.*'):
        for path in ROOT.rglob(pattern):
            if path.is_file() and path not in seen:
                candidates.append(path)
                seen.add(path)
    if not candidates:
        return None
    target = ROOT / 'audit/recovery/stale_atomic_temps' / str(time.time_ns())
    rows = []
    for path in candidates:
        relative = path.relative_to(ROOT)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows.append(dict(path=str(relative), sha256=sha(path), bytes=path.stat().st_size))
        shutil.move(str(path), str(destination))
    event = dict(status='ARCHIVED_UNCOMMITTED_ATOMIC_TEMP_FILES', files=rows, target=str(target), at_unix=time.time())
    write(target / 'EVENT.json', event)
    return event


def integrity(recover_damaged_logs=False):
    source = package_identity()
    if shutil.disk_usage(ROOT).free < 2 * 1024 ** 3:
        raise RuntimeError('INSUFFICIENT_DISK_STOP_WITHOUT_DELETION')
    denied = []
    test_hits = []
    recovered = []
    access = ROOT / 'audit/access'
    if access.exists():
        for path in sorted(access.glob('*.jsonl')):
            valid = []
            error = None
            for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
                try:
                    row = json.loads(raw.decode('utf-8'))
                    valid.append(row)
                    if row.get('permitted') is False:
                        denied.append(dict(path=str(path), line=line_number, row=row))
                    if 'test107' in str(row.get('path', '')).lower():
                        test_hits.append(dict(path=str(path), line=line_number, row=row))
                except Exception as exc:
                    error = dict(line=line_number, error=repr(exc))
                    break
            if error:
                if not recover_damaged_logs:
                    raise RuntimeError('ACCESS_LOG_DAMAGED: ' + str(path))
                recovered.append(archive_damaged_access_log(path, valid, error))
    if denied or test_hits:
        raise RuntimeError('ACCESS_AUDIT_FAILED')
    return dict(
        source_manifest_sha256=source,
        denied_accesses=0,
        test107_text_hits=0,
        recovered_damaged_logs=recovered,
    )


def contains(rows, arm, seed):
    return any(row['arm'] == arm and int(row['seed']) == seed for row in rows)


def valid_training_result(arm, seed):
    folder = run_path(arm, seed)
    result_path = folder / 'RESULT.json'
    hashes_path = folder / 'CHECKPOINT_SHA256.json'
    if not result_path.exists() or not hashes_path.exists():
        return False
    result = read(result_path)
    if result.get('status') != 'COMPLETE_100_EPOCHS':
        return False
    for name, digest in read(hashes_path).items():
        path = folder / name
        if not path.exists() or sha(path) != digest:
            return False
    return True


def valid_evaluation_window(arm, seed):
    path = ROOT / 'evaluation' / arm / f'seed{seed}' / 'WINDOW.json'
    return path.exists() and read(path).get('status') == 'COMPLETE_SIX_CONDITIONS_FIVE_EPOCHS'


def run_schedule(queue, resume):
    for arm, seed in SCHEDULE:
        folder = run_path(arm, seed)
        training_complete = valid_training_result(arm, seed)
        evaluation_complete = valid_evaluation_window(arm, seed)
        committed = (folder / 'COMMITTED_LAST.json').exists()
        folder_has_files = folder.exists() and any(folder.iterdir())

        if training_complete:
            if not contains(queue['started'], arm, seed):
                queue['started'].append(dict(arm=arm, seed=seed, at_unix=time.time(), recovered_registration=True))
            if not contains(queue['completed'], arm, seed):
                queue['completed'].append(dict(arm=arm, seed=seed, recovered_registration=True))
            update(queue, 'B0_S7_100E_RUNNING', 'verify_completed_train', arm, seed)
            integrity()
        elif committed:
            if not resume:
                raise RuntimeError('PARTIAL_RUN_PRESENT_WITHOUT_RESUME_MODE')
            if not contains(queue['started'], arm, seed):
                queue['started'].append(dict(arm=arm, seed=seed, at_unix=time.time(), recovered_registration=True))
            update(queue, 'B0_S7_100E_RUNNING', 'resume_train', arm, seed)
            rc = child('run_candidate.py', ['train', '--arm', arm, '--seed', str(seed), '--resume'], f'train_{arm}_{seed}')
            if rc:
                failures = sorted((ROOT / 'audit/failures').glob(f'train_{arm}_{seed}_*.json'))
                queue['failures'].append(dict(phase='train', arm=arm, seed=seed, exit_code=rc, evidence=[str(path) for path in failures]))
                raise RuntimeError('TRAINING_RUN_FAILED_STOP_ALL')
            if not valid_training_result(arm, seed):
                raise RuntimeError('RESUMED_TRAINING_RESULT_NOT_COMPLETE')
            queue['completed'].append(dict(arm=arm, seed=seed, resumed=True))
            integrity()
        elif folder_has_files:
            raise RuntimeError('NO_COMPLETE_RECOVERY_STATE_REQUIRES_EXPLICIT_RERUN_AUTHORIZATION')
        else:
            if len(queue['started']) >= 6:
                raise RuntimeError('FORMAL_BUDGET_EXCEEDED')
            if contains(queue['started'], arm, seed):
                raise RuntimeError('QUEUE_STARTED_WITHOUT_RUN_ARTIFACTS_REQUIRES_AUDIT')
            queue['started'].append(dict(arm=arm, seed=seed, at_unix=time.time()))
            update(queue, 'B0_S7_100E_RUNNING', 'train', arm, seed)
            rc = child('run_candidate.py', ['train', '--arm', arm, '--seed', str(seed)], f'train_{arm}_{seed}')
            if rc:
                failures = sorted((ROOT / 'audit/failures').glob(f'train_{arm}_{seed}_*.json'))
                queue['failures'].append(dict(phase='train', arm=arm, seed=seed, exit_code=rc, evidence=[str(path) for path in failures]))
                raise RuntimeError('TRAINING_RUN_FAILED_STOP_ALL')
            if not valid_training_result(arm, seed):
                raise RuntimeError('TRAINING_RESULT_NOT_COMPLETE')
            queue['completed'].append(dict(arm=arm, seed=seed))
            integrity()

        if evaluation_complete:
            if not contains(queue['evaluated'], arm, seed):
                queue['evaluated'].append(dict(arm=arm, seed=seed, recovered_registration=True))
            update(queue, 'B0_S7_100E_RUNNING', 'verify_completed_evaluation', arm, seed)
            integrity()
        elif not contains(queue['evaluated'], arm, seed):
            update(queue, 'B0_S7_100E_RUNNING', 'evaluate', arm, seed)
            rc = child('run_candidate.py', ['evaluate', '--arm', arm, '--seed', str(seed)], f'evaluate_{arm}_{seed}')
            if rc:
                failures = sorted((ROOT / 'audit/failures').glob(f'evaluate_{arm}_{seed}_*.json'))
                queue['failures'].append(dict(phase='evaluate', arm=arm, seed=seed, exit_code=rc, evidence=[str(path) for path in failures]))
                raise RuntimeError('EVALUATION_FAILED_PRESERVE_TRAINING_CHECKPOINTS')
            if not valid_evaluation_window(arm, seed):
                raise RuntimeError('EVALUATION_WINDOW_NOT_COMPLETE')
            queue['evaluated'].append(dict(arm=arm, seed=seed))
            integrity()


def initial_queue():
    repair = read(ROOT / 'protocol/REPAIR_PROVENANCE.json')
    return dict(
        protocol=PROTO,
        work_order_sha256=WORK_SHA,
        status='STAGE0_RUNNING',
        pid=os.getpid(),
        schedule=[dict(arm=arm, seed=seed) for arm, seed in SCHEDULE],
        started=[],
        completed=[],
        evaluated=[],
        failures=[],
        resume_events=[],
        prior_failed_attempts=repair['prior_failed_attempts'],
        replacement_rerun_authorized=True,
        total_attempt_budget_including_preserved_failure=7,
        current=None,
        started_unix=time.time(),
        formal_budget=6,
        total_epochs_per_run=100,
        main_window=[96, 97, 98, 99, 100],
        further_stages_authorized=False,
        stagea_stageb_decisions_preserved=True,
    )


def main(resume=False):
    package_identity()
    if resume:
        if not (ROOT / 'QUEUE_STATUS.json').exists():
            raise RuntimeError('NO_QUEUE_TO_RESUME')
        queue = read(ROOT / 'QUEUE_STATUS.json')
        if queue['status'].endswith('ENDED_STOP_PENDING_NEW_AUTHORIZATION'):
            raise RuntimeError('QUEUE_ALREADY_TERMINAL')
    else:
        if (ROOT / 'QUEUE_STATUS.json').exists() or (ROOT / 'SOURCE_AND_PROTOCOL_LOCK.json').exists():
            raise RuntimeError('EXISTING_QUEUE_OR_LOCK_NO_AUTOMATIC_RESTART')
        queue = initial_queue()

    resource_snapshot = start_resource_preflight()
    descriptor, stale_lock = acquire_lock(resume)
    try:
        write(ROOT / 'audit/resource_checks' / f'start_{time.time_ns()}.json', resource_snapshot)
        if resume:
            stale_temps = archive_stale_atomic_temps()
            recovery = integrity(recover_damaged_logs=True)
            event = dict(
                at_unix=time.time(),
                prior_status=queue['status'],
                prior_pid=queue.get('pid'),
                stale_lock=stale_lock,
                stale_atomic_temps=stale_temps,
                integrity=recovery,
            )
            queue.setdefault('resume_events', []).append(event)
            update(queue, 'RESUME_PREFLIGHT_PASS')
        else:
            update(queue, 'STAGE0_RUNNING', 'stage0')

        try:
            if not resume:
                if child('stage0.py', [], 'stage0'):
                    raise RuntimeError('STAGE0_FAILED_STOP_NO_TRAINING')
                update(queue, 'PREDEPLOYMENT_RUNNING', 'predeployment')
                if child('predeployment_checks.py', [], 'predeployment'):
                    raise RuntimeError('PREDEPLOYMENT_FAILED_STOP_NO_TRAINING')
            elif read(ROOT / 'audit/PREDEPLOYMENT_SYNTHETIC.json')['status'] != 'PASS':
                raise RuntimeError('PREDEPLOYMENT_NOT_PASSED_CANNOT_RESUME')

            run_schedule(queue, resume)
            queue['stop_reason'] = 'Authorized matched B0-S7 100-epoch confirmation ended. No later stage was started.'
            update(queue, 'B0_S7_100E_ENDED_STOP_PENDING_NEW_AUTHORIZATION')
        except BaseException as exc:
            queue['stop_reason'] = repr(exc)
            queue['traceback'] = traceback.format_exc()
            update(queue, 'STOPPED_ON_ERROR')
        finally:
            queue['finished_unix'] = time.time()
            try:
                proof = integrity()
                for row in read(ROOT / 'audit/PREFLIGHT_INPUTS.json')['files']:
                    if sha(row['path']) != row['sha256']:
                        raise RuntimeError('FINAL_INPUT_HASH_CHANGED')
                proof['input_hashes_unchanged'] = True
                write(ROOT / 'audit/FINAL_INPUT_SOURCE_CHECK.json', proof)
                result = generate(queue)
                queue.update({key: result[key] for key in ('report_path', 'report_sha256')})
                write(ROOT / 'audit/COMPLETION_AUDIT.json', dict(
                    **result,
                    queue_status=queue['status'],
                    stop_reason=queue['stop_reason'],
                    final_integrity=proof,
                ))
                update(queue, queue['status'])
                paths = [
                    path for path in ROOT.rglob('*')
                    if path.is_file()
                    and path.suffix.lower() in ('.json', '.csv', '.md', '.py', '.patch')
                    and path.name != 'OUTPUT_SHA256.json'
                    and 'runtime_tmp' not in path.parts
                    and 'access' not in path.parts
                ]
                write(ROOT / 'OUTPUT_SHA256.json', [
                    dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)
                    for path in sorted(paths)
                ])
            except BaseException as exc:
                write(ROOT / 'audit/FINALIZATION_ERROR.json', dict(error=repr(exc), traceback=traceback.format_exc()))
    finally:
        os.close(descriptor)
        if GLOBAL_LOCK.exists():
            GLOBAL_LOCK.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.resume)