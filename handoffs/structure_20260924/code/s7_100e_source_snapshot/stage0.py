from __future__ import annotations

import copy
import shutil
import sys
import time

sys.dont_write_bytecode = True
from audit_runtime import (
    ANALYSIS_SHA,
    HASH_LIST_SHA,
    PROTO,
    ROOT,
    STAGEA_CANDIDATE_SHA,
    STAGEA_LOCK_SHA,
    STAGEA_REPORT_SHA,
    STAGEA_ROOT,
    STAGEB_REPORT_SHA,
    STAGEB_ROOT,
    STAGEB_ZIP_SHA,
    WORK_SHA,
    package_identity,
    read,
    sha,
    write,
)
from common import ARMS, BASE_TEMPLATE, EVAL_EPOCHS, EARLY_EPOCHS, SEEDS, TOTAL_EPOCHS, config_path
from runtime_support import check_inputs, environment

ALLOWED_CONFIG_DIFFS = {'SEED', 'EPOCHS'}
SCHEDULE = [(arm, seed) for seed in SEEDS for arm in ARMS]


def changed_keys(before, after):
    keys = set(before) | set(after)
    return sorted(key for key in keys if before.get(key) != after.get(key))


def main():
    source = package_identity()
    for path in (ROOT / 'SOURCE_AND_PROTOCOL_LOCK.json', ROOT / 'RUN_PLAN.json'):
        if path.exists():
            raise RuntimeError('EXISTING_STAGE0_OR_QUEUE_NO_OVERWRITE: ' + str(path))

    seed_check = read(ROOT / 'protocol/SEED_USAGE_CHECK.json')
    if seed_check['status'] != 'PASS' or seed_check['seeds'] != list(SEEDS) or seed_check['true_hits'] != 0:
        raise RuntimeError('NEW_SEED_USAGE_NOT_CLEARED')

    from audit_runtime import REFERENCE_ROOT
    pref = read(REFERENCE_ROOT / 'audit/PREFLIGHT_INPUTS.json')
    if any('test107' in str(row['path']).lower() for row in pref['files']):
        raise RuntimeError('PREFLIGHT_INPUT_LIST_CONTAINS_TEST107')
    write(ROOT / 'audit/PREFLIGHT_INPUTS.json', pref)

    template = read(BASE_TEMPLATE)
    configs = []
    for seed in SEEDS:
        resolved = copy.deepcopy(template)
        resolved['SEED'] = seed
        resolved['EPOCHS'] = TOTAL_EPOCHS
        if set(changed_keys(template, resolved)) != ALLOWED_CONFIG_DIFFS:
            raise RuntimeError('UNDECLARED_CONFIG_DIFFERENCE')
        for arm in ARMS:
            write(config_path(arm, seed), resolved)
            configs.append(dict(arm=arm, seed=seed, path=str(config_path(arm, seed)), changed_keys=changed_keys(template, resolved)))

    stagea_report = STAGEA_ROOT / 'recovery03/reports/FLAME3_4060_SECOND_BATCH_REPORT_RECOVERY03_20260922.md'
    stagea_lock = STAGEA_ROOT / 'BASELINE_REUSE_LOCK.json'
    stagea_candidate = STAGEA_ROOT / 'source_snapshot/candidate_arms.py'
    stageb_report = STAGEB_ROOT / 'reports/FLAME3_S8_GRID_FOLLOWUP_REPORT_20260923.md'
    if sha(stagea_report) != STAGEA_REPORT_SHA:
        raise RuntimeError('STAGEA_REPORT_HASH_MISMATCH')
    if sha(stagea_lock) != STAGEA_LOCK_SHA:
        raise RuntimeError('STAGEA_BASELINE_LOCK_HASH_MISMATCH')
    if sha(stagea_candidate) != STAGEA_CANDIDATE_SHA:
        raise RuntimeError('STAGEA_CANDIDATE_SOURCE_HASH_MISMATCH')
    if sha(stageb_report) != STAGEB_REPORT_SHA:
        raise RuntimeError('STAGEB_REPORT_HASH_MISMATCH')

    source_hash, guard, pref, rt, trainer, factory = __import__('common').setup('stage0', require_lock=False)
    import torch
    check_inputs(pref)
    current_environment = environment(torch)
    expected_environment = read(REFERENCE_ROOT / 'baseline/B0-rep-20260922-v2/seed200/environment.json')
    for key in ('python', 'torch', 'cuda', 'cudnn', 'gpu', 'cudnn_benchmark', 'cudnn_deterministic', 'cudnn_allow_tf32', 'matmul_allow_tf32'):
        if current_environment[key] != expected_environment[key]:
            raise RuntimeError('CURRENT_ENVIRONMENT_MISMATCH: ' + key)
    if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
        raise RuntimeError('INSUFFICIENT_DISK_FOR_SIX_LONG_RUNS')

    run_plan = dict(
        protocol=PROTO,
        status='PASS',
        created_unix=time.time(),
        work_order_sha256=WORK_SHA,
        analysis_sha256=ANALYSIS_SHA,
        hash_list_sha256=HASH_LIST_SHA,
        stageb_zip_sha256=STAGEB_ZIP_SHA,
        source_manifest_sha256=source_hash,
        arms=list(ARMS),
        seeds=list(SEEDS),
        schedule=[dict(arm=arm, seed=seed) for arm, seed in SCHEDULE],
        total_epochs=TOTAL_EPOCHS,
        planned_updates_per_run=6100,
        planned_updates_total=36600,
        main_window=list(EVAL_EPOCHS),
        early_trajectory_window=list(EARLY_EPOCHS),
        conditions=['clean', 'thermal_noise_002', 'thermal_noise_005', 'thermal_noise_010', 'thermal_zero', 'rgb_zero'],
        configs=configs,
        environment=current_environment,
        formal_budget=6,
        next_stage_authorized=False,
        test107_authorized=False,
        audit=guard.snapshot(),
    )
    write(ROOT / 'RUN_PLAN.json', run_plan)
    lock = dict(
        status='PASS',
        protocol=PROTO,
        source_manifest_sha256=source_hash,
        work_order_sha256=WORK_SHA,
        stagea_report=dict(path=str(stagea_report), sha256=STAGEA_REPORT_SHA),
        stagea_baseline_lock=dict(path=str(stagea_lock), sha256=STAGEA_LOCK_SHA),
        stagea_candidate_source=dict(path=str(stagea_candidate), sha256=STAGEA_CANDIDATE_SHA),
        stageb_report=dict(path=str(stageb_report), sha256=STAGEB_REPORT_SHA),
        stageb_delivery_zip=dict(path=str(ROOT / 'protocol/references/FLAME3_S8_STAGEB_20260923_LIGHT_DELIVERY.zip'), sha256=STAGEB_ZIP_SHA),
        seed_usage_check=seed_check,
        config_template=dict(path=str(BASE_TEMPLATE), sha256=sha(BASE_TEMPLATE), allowed_differences=sorted(ALLOWED_CONFIG_DIFFS)),
        input_manifest_sha256=pref['input_manifest_sha256'],
        environment=current_environment,
        same_machine_same_seed_independent_training_repeatability='NOT_ESTABLISHED',
        stagea_and_stageb_decisions_preserved=True,
        no_reference_retraining=True,
    )
    write(ROOT / 'SOURCE_AND_PROTOCOL_LOCK.json', lock)
    write(ROOT / 'INPUT_SOURCE_SHA256.json', dict(
        source_manifest_sha256=source_hash,
        input_manifest_sha256=pref['input_manifest_sha256'],
        approved_inputs=pref['files'],
        references=[lock['stagea_report'], lock['stagea_baseline_lock'], lock['stagea_candidate_source'], lock['stageb_report'], lock['stageb_delivery_zip']],
    ))
    print('STAGE0_PASS: sources, environment, seeds and six-run protocol locked; no training started', flush=True)


if __name__ == '__main__':
    main()