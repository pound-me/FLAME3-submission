from __future__ import annotations

import csv
from datetime import datetime
import json
import statistics
import time

from audit_runtime import ROOT, read, sha, write, write_text
from common import ARMS, EARLY_EPOCHS, EVAL_EPOCHS, SEEDS, run_path

CONDITIONS = ('clean', 'thermal_noise_002', 'thermal_noise_005', 'thermal_noise_010', 'thermal_zero', 'rgb_zero')
KEY = 'smoke_iou_A001_A047'
FP = 'no_fire_joint_false_positive_ratio'


def paired(candidate, baseline):
    clean = candidate['clean'][KEY]
    noise = candidate['thermal_noise_005'][KEY]
    delta_clean = clean - baseline['clean'][KEY]
    delta_noise = noise - baseline['thermal_noise_005'][KEY]
    drop = clean - noise
    baseline_drop = baseline['clean'][KEY] - baseline['thermal_noise_005'][KEY]
    robust = baseline_drop - drop
    fp = candidate['clean'][FP]
    fp_delta = fp - baseline['clean'][FP]
    guards = dict(
        clean_guard=delta_clean >= -.010,
        no_fire_absolute=fp <= .005,
        no_fire_relative=fp_delta <= .001,
    )
    return dict(
        clean_smoke_iou=clean,
        baseline_clean_smoke_iou=baseline['clean'][KEY],
        noise005_smoke_iou=noise,
        baseline_noise005_smoke_iou=baseline['thermal_noise_005'][KEY],
        delta_clean=delta_clean,
        delta_noise005=delta_noise,
        drop=drop,
        baseline_drop=baseline_drop,
        robust_gain=robust,
        no_fire_fp=fp,
        baseline_no_fire_fp=baseline['clean'][FP],
        no_fire_delta=fp_delta,
        **guards,
        precision_component=delta_clean >= .020,
        robust_component=robust >= .030,
        precision_pass=delta_clean >= .020 and guards['no_fire_absolute'] and guards['no_fire_relative'],
        robust_pass=robust >= .030 and all(guards.values()),
    )


def decide(rows):
    if len(rows) != 3 or sorted(row['seed'] for row in rows) != list(SEEDS):
        return dict(status='INCOMPLETE_NOT_JUDGED', precision_axis=None, robust_axis=None)
    precision = all(row['precision_pass'] for row in rows)
    robust = all(row['robust_pass'] for row in rows)
    if precision and robust:
        status = 'PASS_PRECISION_AND_ROBUST_AXES'
    elif precision:
        status = 'PASS_PRECISION_AXIS_ONLY'
    elif robust:
        status = 'PASS_ROBUST_AXIS_ONLY'
    else:
        status = 'COMPLETE_NOT_PASS'
    return dict(
        status=status,
        precision_axis=precision,
        robust_axis=robust,
        all_clean_guards=all(row['clean_guard'] for row in rows),
        all_no_fire_absolute_guards=all(row['no_fire_absolute'] for row in rows),
        all_no_fire_relative_guards=all(row['no_fire_relative'] for row in rows),
        precision_pass_count=sum(row['precision_pass'] for row in rows),
        robust_pass_count=sum(row['robust_pass'] for row in rows),
        clean_guard_count=sum(row['clean_guard'] for row in rows),
        delta_clean_mean=statistics.mean(row['delta_clean'] for row in rows),
        delta_clean_sample_sd=statistics.stdev(row['delta_clean'] for row in rows),
        delta_noise005_mean=statistics.mean(row['delta_noise005'] for row in rows),
        delta_noise005_sample_sd=statistics.stdev(row['delta_noise005'] for row in rows),
        robust_gain_mean=statistics.mean(row['robust_gain'] for row in rows),
        robust_gain_sample_sd=statistics.stdev(row['robust_gain'] for row in rows),
        positive_clean_seeds=sum(row['delta_clean'] > 0 for row in rows),
        positive_noise005_seeds=sum(row['delta_noise005'] > 0 for row in rows),
        positive_robust_seeds=sum(row['robust_gain'] > 0 for row in rows),
        thresholds=dict(clean_guard=-.010, no_fire_absolute=.005, no_fire_relative_delta=.001, precision_delta=.020, robust_gain=.030),
    )


def csvout(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [
            dict(field=key, value_json=json.dumps(value, ensure_ascii=False, sort_keys=True))
            for key, value in rows.items()
        ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['status'])
        writer.writeheader()
        writer.writerows(rows)


def current(queue):
    lines = [
        '# FLAME3 4060Ti B0-S7 100-Epoch Current Status',
        '',
        'Updated: ' + datetime.now().astimezone().isoformat(),
        'Platform: DESKTOP-FNGL8MM / RTX4060Ti 16GB',
        'Protocol: FLAME3_4060TI_B0_S7_100E_20260924_V1',
        '',
        'Status: ' + queue['status'],
        'Current: ' + json.dumps(queue.get('current'), ensure_ascii=False),
        f"Formal: started {len(queue['started'])}/6; completed {len(queue['completed'])}; evaluated {len(queue['evaluated'])}; failures {len(queue['failures'])}.",
        f"Preserved prior failed attempts: {len(queue.get('prior_failed_attempts', []))}.",
        'Schedule: B0-203, S7-203, B0-204, S7-204, B0-205, S7-205.',
        'Main window: epochs 96-100. Epochs 26-30 are trajectory-only.',
        'Stage A/B conclusions: preserved unchanged.',
        'No test107 access. No S8/S2/S5/R1/N/extra seed/extra grid.',
        '',
        'Stop/next: ' + queue.get('stop_reason', 'Continue only through the authorized six runs and fixed evaluation, then stop.'),
        'Final report: ' + queue.get('report_path', 'NOT_CREATED_YET'),
        'Final report SHA256: ' + queue.get('report_sha256', 'NOT_CREATED_YET'),
    ]
    write_text(ROOT / 'CURRENT_STATUS.md', '\n'.join(lines) + '\n')


def generate(queue):
    windows = []
    fp_rows = []
    paired_rows = []
    checkpoints = []
    trajectory = []
    registry = dict(
        formal_started=queue['started'],
        formal_completed=queue['completed'],
        formal_evaluated=queue['evaluated'],
        failures=queue['failures'],
        resume_events=queue.get('resume_events', []),
        prior_failed_attempts=queue.get('prior_failed_attempts', []),
        schedule=queue['schedule'],
    )
    by_arm_seed = {}
    for arm in ARMS:
        for seed in SEEDS:
            window_path = ROOT / 'evaluation' / arm / f'seed{seed}' / 'WINDOW.json'
            if window_path.exists():
                window = read(window_path)['window']
                by_arm_seed[(arm, seed)] = window
                for condition, metrics in window.items():
                    windows.append(dict(arm=arm, seed=seed, condition=condition, **metrics))
            folder = run_path(arm, seed)
            hashes = folder / 'CHECKPOINT_SHA256.json'
            if hashes.exists():
                for name, digest in read(hashes).items():
                    path = folder / name
                    checkpoints.append(dict(arm=arm, seed=seed, path=str(path), name=name, sha256=digest, bytes=path.stat().st_size))
            for epoch in (*EARLY_EPOCHS, *EVAL_EPOCHS):
                path = folder / 'epochs' / f'epoch{epoch:03d}.json'
                if path.exists():
                    row = read(path)
                    view = row['validation']
                    trajectory.append(dict(
                        arm=arm,
                        seed=seed,
                        epoch=epoch,
                        role='MAIN_WINDOW' if epoch in EVAL_EPOCHS else 'EARLY_TRAJECTORY_RECORD_ONLY',
                        train_loss=row['train_loss'],
                        validation_loss=row['validation_loss'],
                        smoke_iou_A001_A047=view[KEY],
                        no_fire_joint_false_positive_ratio=view[FP],
                        lr_end=row['lr_end'][0],
                        scaler_scale=row['scaler']['scale'],
                        scaler_skipped_steps=row['scaler_skipped_steps'],
                    ))

    for seed in SEEDS:
        if ('B0', seed) in by_arm_seed and ('S7', seed) in by_arm_seed:
            baseline = by_arm_seed[('B0', seed)]
            candidate = by_arm_seed[('S7', seed)]
            row = dict(seed=seed, **paired(candidate, baseline))
            row.update(
                thermal_zero_fp=candidate['thermal_zero'][FP],
                baseline_thermal_zero_fp=baseline['thermal_zero'][FP],
                thermal_zero_fp_delta=candidate['thermal_zero'][FP] - baseline['thermal_zero'][FP],
                rgb_zero_smoke_iou=candidate['rgb_zero'][KEY],
                baseline_rgb_zero_smoke_iou=baseline['rgb_zero'][KEY],
                rgb_zero_smoke_delta=candidate['rgb_zero'][KEY] - baseline['rgb_zero'][KEY],
            )
            paired_rows.append(row)
            for arm, window in [('B0', baseline), ('S7', candidate)]:
                for condition, metrics in window.items():
                    fp_rows.append(dict(
                        arm=arm,
                        seed=seed,
                        condition=condition,
                        absolute_no_fire_fp=metrics[FP],
                        delta_fp_vs_b0=metrics[FP] - baseline[condition][FP],
                    ))

    decision = decide(paired_rows)
    precision_decision = dict(
        status='PASS' if decision.get('precision_axis') else 'FAIL' if decision.get('precision_axis') is False else 'INCOMPLETE',
        axis='precision',
        rule='All three seeds require delta_clean >= 0.020 and both normal-input No-Fire guards.',
        rows=[{key: row[key] for key in ('seed', 'delta_clean', 'no_fire_fp', 'no_fire_delta', 'precision_pass')} for row in paired_rows],
    )
    robust_decision = dict(
        status='PASS' if decision.get('robust_axis') else 'FAIL' if decision.get('robust_axis') is False else 'INCOMPLETE',
        axis='robust',
        rule='All three seeds require robust_gain >= 0.030 plus clean and normal-input No-Fire guards.',
        rows=[{key: row[key] for key in ('seed', 'delta_clean', 'robust_gain', 'clean_guard', 'no_fire_absolute', 'no_fire_relative', 'robust_pass')} for row in paired_rows],
    )

    summaries = []
    for arm in ARMS:
        for condition in CONDITIONS:
            selected = [row for row in windows if row['arm'] == arm and row['condition'] == condition]
            if len(selected) == 3:
                for key in selected[0]:
                    if key in ('arm', 'seed', 'condition'):
                        continue
                    summaries.append(dict(
                        arm=arm,
                        condition=condition,
                        metric=key,
                        mean=statistics.mean(row[key] for row in selected),
                        sample_sd=statistics.stdev(row[key] for row in selected),
                    ))

    artifacts = [
        ('RUN_REGISTRY', registry),
        ('CHECKPOINT_MANIFEST', checkpoints),
        ('WINDOW_METRICS', windows),
        ('WINDOW_SUMMARY', summaries),
        ('PAIRED_DELTAS', paired_rows),
        ('ALL_CONDITION_FP', fp_rows),
        ('EARLY_LATE_TRAJECTORY', trajectory),
    ]
    for name, value in artifacts:
        write(ROOT / (name + '.json'), value)
        csvout(ROOT / (name + '.csv'), value)
    write(ROOT / 'PRECISION_DECISION.json', precision_decision)
    write(ROOT / 'ROBUST_DECISION.json', robust_decision)
    write(ROOT / 'DECISIONS.json', dict(
        overall=decision,
        precision=precision_decision,
        robust=robust_decision,
        next_stage_authorized=False,
        stagea_stageb_decisions_preserved=True,
    ))

    def value(key):
        return 'NA' if key not in decision else f'{100 * decision[key]:+.4f}'

    lines = [
        '# FLAME3 B0-S7 Matched 100-Epoch Confirmation Report',
        '',
        'Date: 2026-09-24',
        'Status: ' + queue['status'],
        'Stop reason: ' + queue.get('stop_reason', 'Authorized six-run confirmation ended.'),
        '',
        '## Execution',
        '',
        f'- Formal runs started/completed/evaluated: {len(queue["started"])}/{len(queue["completed"])}/{len(queue["evaluated"])}; authorized maximum 6.',
        f'- Failure events in repair run: {len(queue["failures"])}. Resume events: {len(queue.get("resume_events", []))}.',
        f'- Preserved prior failed attempts: {len(queue.get("prior_failed_attempts", []))}; these are not erased from the experiment registry.',
        '- Models: matched B0 and exact Stage A S7; seeds 203/204/205; 100 epochs each; poly100.',
        '- Main endpoint window: epochs 96-100. Epochs 26-30 are trajectory-only and were not used for checkpoint selection.',
        '- This remains a repeatedly used development-set experiment; new seeds are not a new independent dataset.',
        '',
        '## Overall Decision',
        '',
        f'- Status: {decision["status"]}.',
        f'- Precision axis: {decision["precision_axis"]}. Robust axis: {decision["robust_axis"]}.',
        f'- Mean clean delta: {value("delta_clean_mean")} pp.',
        f'- Mean sigma=.05 absolute delta: {value("delta_noise005_mean")} pp.',
        f'- Mean degradation improvement R: {value("robust_gain_mean")} pp.',
        '- Thresholds are project utility gates, not statistical significance claims.',
        '',
        '## Same-Seed Paired Results',
        '',
        '| Seed | B0 clean (%) | S7 clean (%) | Delta clean (pp) | Delta noise .05 (pp) | R (pp) | Clean guard | FP abs | FP relative | Precision pass | Robust pass |',
        '|---:|---:|---:|---:|---:|---:|---|---|---|---|---|',
    ]
    for row in paired_rows:
        lines.append(
            f'| {row["seed"]} | {100*row["baseline_clean_smoke_iou"]:.4f} | {100*row["clean_smoke_iou"]:.4f} | '
            f'{100*row["delta_clean"]:+.4f} | {100*row["delta_noise005"]:+.4f} | {100*row["robust_gain"]:+.4f} | '
            f'{row["clean_guard"]} | {row["no_fire_absolute"]} | {row["no_fire_relative"]} | {row["precision_pass"]} | {row["robust_pass"]} |'
        )
    lines += [
        '',
        '## Modality-Zero Side Effects',
        '',
        '| Seed | S7 thermal-zero FP (%) | Delta vs B0 (pp) | S7 rgb-zero Smoke IoU (%) | Delta vs B0 (pp) |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in paired_rows:
        lines.append(
            f'| {row["seed"]} | {100*row["thermal_zero_fp"]:.4f} | {100*row["thermal_zero_fp_delta"]:+.4f} | '
            f'{100*row["rgb_zero_smoke_iou"]:.4f} | {100*row["rgb_zero_smoke_delta"]:+.4f} |'
        )
    lines += [
        '',
        '## Evidence and Boundary',
        '',
        '- Fire/Heat, Background, mIoU and S are record-only.',
        '- No best-checkpoint, per-seed model choice, early stopping or threshold change was used.',
        '- No test107 image, label, prediction, statistic or sample list was accessed.',
        '- Original Stage A and Stage B decisions remain unchanged.',
        '- Next stage approved: NO. Further training requires a new explicit authorization.',
    ]
    terminal = queue['status'] == 'B0_S7_100E_ENDED_STOP_PENDING_NEW_AUTHORIZATION'
    if terminal:
        report = ROOT / 'reports/B0_S7_100E_CONFIRMATION_REPORT_20260923.md'
        if report.exists():
            raise RuntimeError('FINAL_REPORT_ALREADY_EXISTS')
    else:
        report = ROOT / 'reports' / f'INTERIM_STOP_REPORT_{time.time_ns()}.md'
    write_text(report, '\n'.join(lines) + '\n')
    return dict(
        report_path=str(report),
        report_sha256=sha(report),
        formal_started=len(queue['started']),
        formal_completed=len(queue['completed']),
        formal_evaluated=len(queue['evaluated']),
        failure_events=len(queue['failures']),
        resume_events=len(queue.get('resume_events', [])),
        decision=decision,
        next_stage_authorized=False,
    )