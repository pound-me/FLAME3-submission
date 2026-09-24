from __future__ import annotations

import argparse
import importlib.util
import sys
import time

sys.dont_write_bytecode = True
from audit_runtime import PERTURB_SHA, PROTO, REFERENCE_ROOT, ROOT, package_identity, read, sha, write
from checkpointing import capture_rng, commit_epoch, load_committed, restore_rng
from common import (
    ARMS,
    EARLY_EPOCHS,
    EVAL_EPOCHS,
    SAVE_EPOCHS,
    SEEDS,
    TOTAL_EPOCHS,
    CountedLoader,
    bn_counters,
    bn_deltas,
    config_path,
    fail,
    prepare,
    run_path,
    setup,
)
from runtime_support import check_inputs, environment, tensor_state_sha


def train(arm, seed, resume=False):
    source, guard, pref, rt, trainer, factory = setup(f'train_{arm}_{seed}')
    import torch
    from stage3_protocol import assert_finite, average_views, metric_view

    if read(ROOT / 'audit/PREDEPLOYMENT_SYNTHETIC.json')['status'] != 'PASS':
        raise RuntimeError('PREDEPLOYMENT_CONTRACTS_NOT_PASSED')
    folder = run_path(arm, seed)
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise RuntimeError('EXISTING_RUN_NO_AUTOMATIC_RESTART')
    folder.mkdir(parents=True, exist_ok=True)
    values = prepare(rt, trainer, factory, arm, seed, verify_expected=True)
    config, model, criterion, optimizer, scaler = values[:5]
    train_generator, validation_generator, training, validation, targets, initial = values[5:]
    expected = dict(
        protocol=PROTO,
        arm=arm,
        seed=seed,
        config=config,
        source_manifest_sha256=source,
        input_manifest_sha256=pref['input_manifest_sha256'],
        engineering_only=False,
    )
    records = []
    start = 0
    if resume:
        saved, pointer, archive_event = load_committed(torch, folder, expected, SAVE_EPOCHS)
        start = int(saved['epoch'])
        records = [read(folder / 'epochs' / f'epoch{epoch:03d}.json') for epoch in range(1, start + 1)]
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict'])
        restore_rng(torch, saved['rng'], train_generator, validation_generator)
        write(folder / f'RESUME_{time.time_ns()}.json', dict(
            epoch=start,
            committed=pointer,
            archived_uncommitted_tail=archive_event,
            same_run=True,
            score_triggered=False,
            protocol_changed=False,
        ))
        del saved
    else:
        write(folder / 'resolved_config.json', config)
        write(folder / 'INITIALIZATION.json', initial)
        write(folder / 'environment.json', dict(**environment(torch), deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))

    steps = []

    def step_observer(opt, args, kwargs):
        steps.append(dict(
            update=len(steps) + 1,
            lr=[group['lr'] for group in opt.param_groups],
            scaler_before_update=scaler.get_scale(),
        ))

    hook = optimizer.register_step_post_hook(step_observer)
    for epoch_index in range(start, TOTAL_EPOCHS):
        epoch = epoch_index + 1
        guard.rotate(f'train_{arm}_{seed}_epoch{epoch:03d}')
        write(folder / 'STATUS.json', dict(
            status='TRAINING',
            arm=arm,
            seed=seed,
            current_epoch=epoch,
            completed_epochs=epoch_index,
            total_epochs=TOTAL_EPOCHS,
            updated_unix=time.time(),
        ))
        print(f'train {arm} seed{seed} epoch{epoch}/{TOTAL_EPOCHS}', flush=True)
        steps.clear()
        bn_before = bn_counters(model)
        began = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        counted = CountedLoader(training, trace=True)
        train_metrics, loss, components, _, _ = trainer.run_training_epoch(
            model,
            counted,
            criterion,
            optimizer,
            scaler,
            config,
            epoch_index,
            TOTAL_EPOCHS,
            None,
            True,
        )
        train_seconds = time.perf_counter() - began
        if (counted.batches, counted.items) != (61, 488):
            raise RuntimeError('INCOMPLETE_TRAINING_EPOCH')
        pairing = counted.proof()
        if arm == 'S7':
            reference_path = run_path('B0', seed) / 'epochs' / f'epoch{epoch:03d}.json'
            if not reference_path.exists():
                raise RuntimeError('PAIRED_B0_EPOCH_NOT_AVAILABLE')
            baseline_pairing = read(reference_path)['data_pairing']
            pairing_exact = pairing == baseline_pairing
            if not pairing_exact:
                raise RuntimeError('B0_S7_DATA_ORDER_OR_GEOMETRY_MISMATCH')
        else:
            baseline_pairing = pairing
            pairing_exact = True
        if len(steps) != 61:
            raise RuntimeError('NONFINITE_OR_SKIPPED_OPTIMIZER_UPDATE')
        if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
            raise RuntimeError('NONFINITE_MODEL_STATE')
        updates = bn_deltas(bn_before, bn_counters(model), 61)

        before_eval = tensor_state_sha(model.state_dict())
        validate_started = time.perf_counter()
        val_counted = CountedLoader(validation)
        validation_metrics, validation_loss, validation_components, _ = trainer.run_validation(
            model,
            val_counted,
            criterion,
            config,
            epoch_index,
            None,
            True,
            targets,
        )
        if (val_counted.batches, val_counted.items) != (17, 134):
            raise RuntimeError('INCOMPLETE_VALIDATION_EPOCH')
        if tensor_state_sha(model.state_dict()) != before_eval:
            raise RuntimeError('VALIDATION_CHANGED_MODEL_OR_BN_STATE')

        record = dict(
            epoch=epoch,
            arm=arm,
            seed=seed,
            train_loss=loss,
            validation_loss=validation_loss,
            train=train_metrics,
            validation=validation_metrics,
            loss_components=dict(train=components, validation=validation_components),
            train_batches=61,
            train_items=488,
            validation_batches=17,
            validation_items=134,
            data_pairing=pairing,
            paired_b0_data_pairing=baseline_pairing,
            baseline_data_pairing_exact=pairing_exact,
            lr_end=[group['lr'] for group in optimizer.param_groups],
            lr_total_epochs=100,
            lr_total_steps=6100,
            completed_global_steps=epoch * 61,
            optimizer_updates=len(steps),
            scaler_skipped_steps=61 - len(steps),
            optimizer_step_records=list(steps),
            scaler=scaler.state_dict(),
            bn_updates=updates,
            train_generator_state_sha256=tensor_state_sha({'train_generator': train_generator.get_state()}),
            validation_generator_state_sha256=tensor_state_sha({'validation_generator': validation_generator.get_state()}),
            train_seconds=train_seconds,
            validation_seconds=time.perf_counter() - validate_started,
            epoch_elapsed_seconds=time.perf_counter() - began,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
            S_use='RECORD_ONLY',
            checkpoint_selection='FIXED_WINDOW_96_100',
            early_trajectory_record_only=epoch in EARLY_EPOCHS,
            validation_model_bn_unchanged=True,
        )
        assert_finite(record)
        payload = dict(
            **expected,
            epoch=epoch,
            global_step=epoch * 61,
            scheduler_state=dict(kind='polynomial_function', total_steps=6100, power=.9, last_step=epoch * 61 - 1),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict=scaler.state_dict(),
            rng=capture_rng(torch, train_generator, validation_generator),
            epoch_record=record,
        )
        pointer = commit_epoch(torch, folder, payload, record, SAVE_EPOCHS)
        records.append(record)
        write(folder / 'STATUS.json', dict(
            status='EPOCH_COMPLETE',
            completed_epochs=epoch,
            total_epochs=TOTAL_EPOCHS,
            arm=arm,
            seed=seed,
            committed=pointer,
            updated_unix=time.time(),
            elapsed_seconds=record['epoch_elapsed_seconds'],
        ))
    hook.remove()

    check_inputs(pref)
    if package_identity() != source:
        raise RuntimeError('SOURCE_CHANGED_DURING_TRAINING')
    names = ['last.pth', *[f'epoch{epoch:03d}.pth' for epoch in SAVE_EPOCHS]]
    hashes = {name: sha(folder / name) for name in names}
    write(folder / 'CHECKPOINT_SHA256.json', hashes)
    result = dict(
        status='COMPLETE_100_EPOCHS',
        arm=arm,
        seed=seed,
        epochs=TOTAL_EPOCHS,
        early_window=average_views([metric_view(records[epoch - 1]['validation']) for epoch in EARLY_EPOCHS]),
        clean_window=average_views([metric_view(records[epoch - 1]['validation']) for epoch in EVAL_EPOCHS]),
        checkpoint_sha256=hashes,
        source_manifest_sha256=source,
        input_hashes_unchanged=True,
        audit=guard.snapshot(),
    )
    write(folder / 'RESULT.json', result)
    write(folder / 'STATUS.json', dict(status='COMPLETE_100_EPOCHS', completed_epochs=100, total_epochs=100, arm=arm, seed=seed))


def evaluate(arm, seed):
    source, guard, pref, rt, trainer, factory = setup(f'evaluate_{arm}_{seed}')
    import torch
    from torch.utils.data import DataLoader
    from candidate_arms import apply_arm
    from engineering_checks import make_criterion
    from run_stage3 import PerturbedLoader
    from stage3_protocol import CONDITIONS, assert_finite, average_views, metric_view

    folder = run_path(arm, seed)
    result = read(folder / 'RESULT.json')
    if result['status'] != 'COMPLETE_100_EPOCHS':
        raise RuntimeError('INCOMPLETE_TRAINING_CANNOT_EVALUATE')
    perturb_path = REFERENCE_ROOT / 'frozen/perturb_source.py'
    if sha(perturb_path) != PERTURB_SHA:
        raise RuntimeError('PERTURB_SOURCE_MISMATCH')
    spec = importlib.util.spec_from_file_location('frozen_pure_perturb', perturb_path)
    perturb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perturb)
    rows = []
    for epoch in EVAL_EPOCHS:
        guard.rotate(f'evaluate_{arm}_{seed}_epoch{epoch:03d}')
        checkpoint = folder / f'epoch{epoch:03d}.pth'
        digest = sha(checkpoint)
        if digest != result['checkpoint_sha256'][checkpoint.name]:
            raise RuntimeError('CHECKPOINT_HASH_MISMATCH')
        output = ROOT / 'evaluation' / arm / f'seed{seed}' / f'epoch{epoch:03d}.json'
        if output.exists():
            raise RuntimeError('EXISTING_EVALUATION_NO_OVERWRITE')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = read(config_path(arm, seed))
        expected = dict(
            protocol=PROTO,
            arm=arm,
            seed=seed,
            epoch=epoch,
            config=config,
            source_manifest_sha256=source,
            input_manifest_sha256=pref['input_manifest_sha256'],
            engineering_only=False,
        )
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError('EVALUATION_CHECKPOINT_IDENTITY_MISMATCH')
        dataset = rt.build_dataset(config, 'val')
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        model = apply_arm(rt.build_model(config, augment=True), arm, seed)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model = model.cuda().eval()
        criterion = make_criterion(rt, factory, config, 'baseline')
        targets = trainer.load_frozen_manual_dev_targets(config['FLAME3_MANUAL_SMOKE_V2_ANNOTATION_PACKAGE'])
        expected_clean = metric_view(read(folder / 'epochs' / f'epoch{epoch:03d}.json')['validation'])
        before = tensor_state_sha(model.state_dict())
        row = dict(
            arm=arm,
            seed=seed,
            epoch=epoch,
            checkpoint_sha256=digest,
            source_manifest_sha256=source,
            input_manifest_sha256=pref['input_manifest_sha256'],
            perturb_source_sha256=PERTURB_SHA,
            conditions={},
            condition_seconds={},
            evaluation_random_stream='SEPARATE_DETERMINISTIC_PERTURBATION',
        )
        for condition in CONDITIONS:
            started = time.perf_counter()
            counted = CountedLoader(PerturbedLoader(loader, trainer, perturb.perturb, condition, torch.device('cuda:0')))
            metrics, _, _, _ = trainer.run_validation(model, counted, criterion, config, epoch - 1, None, True, targets)
            if (counted.batches, counted.items) != (17, 134):
                raise RuntimeError('INCOMPLETE_CONDITION_EVALUATION')
            row['conditions'][condition['id']] = metric_view(metrics)
            row['condition_seconds'][condition['id']] = time.perf_counter() - started
        row['model_and_bn_unchanged'] = before == tensor_state_sha(model.state_dict())
        row['clean_replay_max_abs'] = max(abs(row['conditions']['clean'][key] - value) for key, value in expected_clean.items())
        row['clean_replay_contract_pass'] = row['clean_replay_max_abs'] < 1e-6
        assert_finite(row)
        write(output, row)
        if not row['model_and_bn_unchanged'] or not row['clean_replay_contract_pass']:
            raise RuntimeError('EVALUATION_READONLY_OR_REPLAY_FAILED')
        rows.append(row)
        del model, payload, criterion
        torch.cuda.empty_cache()

    check_inputs(pref)
    if package_identity() != source:
        raise RuntimeError('SOURCE_CHANGED_DURING_EVALUATION')
    write(ROOT / 'evaluation' / arm / f'seed{seed}' / 'WINDOW.json', dict(
        status='COMPLETE_SIX_CONDITIONS_FIVE_EPOCHS',
        arm=arm,
        seed=seed,
        epochs=list(EVAL_EPOCHS),
        window={condition['id']: average_views([row['conditions'][condition['id']] for row in rows]) for condition in CONDITIONS},
        input_hashes_unchanged=True,
        model_and_bn_unchanged=True,
        audit=guard.snapshot(),
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'evaluate'])
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        if args.mode == 'evaluate':
            evaluate(args.arm, args.seed)
        else:
            train(args.arm, args.seed, args.resume)
    except BaseException as exc:
        fail(args.mode, args.arm, args.seed, exc)
        raise