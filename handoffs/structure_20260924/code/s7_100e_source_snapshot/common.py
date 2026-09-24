from __future__ import annotations

import hashlib
import json
import random
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
from audit_runtime import ROOT, REFERENCE_ROOT, package_identity, read, sha, write
from runtime_support import bootstrap, check_inputs, environment, set_environment, tensor_state_sha

ARMS = ('B0', 'S7')
SEEDS = (203, 204, 205)
TOTAL_EPOCHS = 100
EARLY_EPOCHS = (26, 27, 28, 29, 30)
EVAL_EPOCHS = (96, 97, 98, 99, 100)
SAVE_EPOCHS = EARLY_EPOCHS + EVAL_EPOCHS
BASE_TEMPLATE = REFERENCE_ROOT / 'baseline/B0-rep-20260922-v2/seed200/resolved_config.json'


def config_path(arm, seed):
    return ROOT / 'RESOLVED_CONFIGS' / arm / f'seed{seed}.json'


def run_path(arm, seed):
    return ROOT / 'formal' / arm / f'seed{seed}'


def setup(phase, require_lock=True):
    source_hash = package_identity()
    guard, pref, rt, trainer, factory = bootstrap(phase)
    import torch
    expected = read(ROOT / 'RUN_PLAN.json')['environment'] if (ROOT / 'RUN_PLAN.json').exists() else read(REFERENCE_ROOT / 'baseline/B0-rep-20260922-v2/seed200/environment.json')
    set_environment(torch, expected)
    check_inputs(pref)
    if require_lock:
        lock = read(ROOT / 'SOURCE_AND_PROTOCOL_LOCK.json')
        if lock['status'] != 'PASS':
            raise RuntimeError('SOURCE_AND_PROTOCOL_NOT_LOCKED')
        if lock['source_manifest_sha256'] != source_hash:
            raise RuntimeError('SOURCE_CHANGED_AFTER_PROTOCOL_LOCK')
    return source_hash, guard, pref, rt, trainer, factory


def worker_guard(worker_id):
    from audit_runtime import AccessAudit
    global _worker_audit
    _worker_audit = AccessAudit(read(ROOT / 'audit/PREFLIGHT_INPUTS.json')['allowed'], 'worker_' + str(worker_id))


class CountedLoader:
    def __init__(self, loader, trace=False):
        self.loader, self.trace = loader, trace
        self.batches, self.items = 0, 0
        self.order, self.geometry = hashlib.sha256(), hashlib.sha256()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.batches += 1
            self.items += len(batch[0])
            if self.trace:
                value = batch[3]
                if len(value) == 1 and isinstance(value[0], (tuple, list)):
                    value = value[0]
                self.order.update(json.dumps([str(item) for item in value]).encode())
                for item in batch[:3]:
                    self.geometry.update(item.detach().contiguous().numpy().tobytes())
            yield batch

    def proof(self):
        return dict(
            batches=self.batches,
            items=self.items,
            sample_order_sha256=self.order.hexdigest().upper(),
            pre_thermal_geometry_sha256=self.geometry.hexdigest().upper(),
        )


def _rng_snapshot(torch):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state().clone(),
        cuda=[state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None,
    )


def _rng_equal(torch, before, after):
    numpy_equal = before['numpy'][0] == after['numpy'][0] and np.array_equal(before['numpy'][1], after['numpy'][1]) and before['numpy'][2:] == after['numpy'][2:]
    cuda_equal = (before['cuda'] is None and after['cuda'] is None) or (
        before['cuda'] is not None and after['cuda'] is not None and
        len(before['cuda']) == len(after['cuda']) and
        all(torch.equal(a, b) for a, b in zip(before['cuda'], after['cuda']))
    )
    return before['python'] == after['python'] and numpy_equal and torch.equal(before['cpu'], after['cpu']) and cuda_equal


def prepare(rt, trainer, factory, arm, seed, verify_expected=True):
    import torch
    from torch.utils.data import DataLoader
    from engineering_checks import make_criterion
    from candidate_arms import apply_arm, assert_unchanged_state

    if arm not in ARMS or seed not in SEEDS:
        raise RuntimeError('UNAUTHORIZED_ARM_OR_SEED')
    config = read(config_path(arm, seed))
    paired = read(config_path('B0', seed))
    if config != paired:
        raise RuntimeError('B0_S7_CONFIG_MISMATCH')
    rt.seed_everything(seed)
    training = rt.build_dataset(config, 'train')
    validation = rt.build_dataset(config, 'val')
    targets = trainer.load_frozen_manual_dev_targets(config['FLAME3_MANUAL_SMOKE_V2_ANNOTATION_PACKAGE'])
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 100000)
    if trainer.build_flame3_train_sampler(training, config, train_generator) is not None:
        raise RuntimeError('UNDECLARED_TRAIN_SAMPLER')
    train_loader = DataLoader(
        training,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        generator=train_generator,
        worker_init_fn=worker_guard,
    )
    val_loader = DataLoader(
        validation,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        generator=validation_generator,
        worker_init_fn=worker_guard,
    )
    if (len(training), len(validation), len(train_loader), len(val_loader)) != (493, 134, 61, 17):
        raise RuntimeError('DATA_INTERFACE_CHANGED')

    baseline = rt.build_model(config, augment=True)
    stem = baseline.conv1[0].weight.detach().clone()
    matched = rt.load_pretrained_if_available(baseline, config)
    baseline_state = tensor_state_sha(baseline.state_dict())
    before = _rng_snapshot(torch)
    model = apply_arm(baseline, arm, seed)
    after = _rng_snapshot(torch)
    if not _rng_equal(torch, before, after):
        raise RuntimeError('CANDIDATE_CONSTRUCTION_CHANGED_RNG')
    unchanged = assert_unchanged_state(baseline, model, arm)
    model_state = tensor_state_sha(model.state_dict())
    stem_sha = tensor_state_sha({'stem': stem})
    initial = dict(
        seed=seed,
        arm=arm,
        model_state_sha256=model_state,
        baseline_state_sha256=baseline_state,
        stem_sha256=stem_sha,
        matched_pretrained=matched,
        common_state=unchanged,
        all_rng_preserved=True,
        prior_checkpoint_loaded=False,
        formal_from_seed_initialization=True,
    )
    if matched != 301 or not torch.equal(stem, model.conv1[0].weight):
        raise RuntimeError('PRETRAINED_OR_STEM_INITIALIZATION_MISMATCH')
    if verify_expected:
        expected = read(ROOT / 'INITIALIZATION_CHECK/EXPECTED_INITIALIZATION.json')['seeds'][str(seed)]
        for key in ('baseline_state_sha256', 'stem_sha256'):
            if initial[key] != expected[key]:
                raise RuntimeError('INITIALIZATION_IDENTITY_MISMATCH: ' + key)
        if initial['model_state_sha256'] != expected['arms'][arm]['model_state_sha256']:
            raise RuntimeError('ARM_INITIALIZATION_IDENTITY_MISMATCH')
    del baseline
    model = model.cuda()
    criterion = make_criterion(rt, factory, config, 'baseline')
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['LR'],
        momentum=config['MOMENTUM'],
        weight_decay=config['WD'],
        dampening=0,
        nesterov=False,
    )
    covered = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    if len(covered) != len(set(covered)) or set(covered) != {id(parameter) for parameter in model.parameters()}:
        raise RuntimeError('OPTIMIZER_PARAMETER_COVERAGE_FAILED')
    scaler = trainer.create_cuda_grad_scaler(enabled=True, init_scale=config['AMP_INIT_SCALE'])
    return (
        config,
        model,
        criterion,
        optimizer,
        scaler,
        train_generator,
        validation_generator,
        train_loader,
        val_loader,
        targets,
        initial,
    )


def bn_counters(model):
    return {key: int(value) for key, value in model.state_dict().items() if key.endswith('num_batches_tracked')}


def bn_deltas(before, after, expected):
    delta = {key: after[key] - value for key, value in before.items()}
    if set(before) != set(after) or any(value != expected for value in delta.values()):
        raise RuntimeError('BN_UPDATE_COUNT_CHANGED')
    return delta


def fail(phase, arm, seed, exc):
    import traceback
    text = str(exc).lower()
    resource = 'out of memory' in text or 'insufficient' in text
    numeric = any(word in text for word in ('non-finite', 'nonfinite', 'non finite', 'numeric_gate_failed'))
    kind = 'RESOURCE_BLOCKED' if resource else 'NUMERICAL_ARM_FAILURE' if numeric else 'SHARED_OR_IMPLEMENTATION_FAILURE'
    record = dict(
        status='FAILED_STOP',
        kind=kind,
        phase=phase,
        arm=arm,
        seed=seed,
        error=repr(exc),
        traceback=traceback.format_exc(),
        at_unix=time.time(),
    )
    write(ROOT / 'audit/failures' / f'{phase}_{arm}_{seed}_{time.time_ns()}.json', record)
    return record