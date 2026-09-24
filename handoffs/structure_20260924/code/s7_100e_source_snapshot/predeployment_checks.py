from __future__ import annotations

import copy
import importlib.util
import math
import sys

sys.dont_write_bytecode = True
from audit_runtime import ROOT, STAGEA_CANDIDATE_SHA, STAGEA_ROOT, package_identity, read, sha, write
from checkpointing import capture_rng, restore_rng, save_torch_verified
from common import ARMS, SEEDS, config_path, prepare, setup
from runtime_support import tensor_state_sha


def direct_stagea_module():
    path = STAGEA_ROOT / 'source_snapshot/candidate_arms.py'
    if sha(path) != STAGEA_CANDIDATE_SHA:
        raise RuntimeError('DIRECT_STAGEA_CANDIDATE_HASH_MISMATCH')
    spec = importlib.util.spec_from_file_location('direct_stagea_candidate_arms', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scheduler_contract(torch, trainer):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=.001, momentum=.9, weight_decay=1e-5)
    rows = []
    for step in (0, 1829, 1830, 5855, 6099):
        actual = trainer.set_polynomial_lr(optimizer, .001, step, 6100, .9)
        expected = max(.001 * (1.0 - step / 6100) ** .9, 1e-8)
        rows.append(dict(step=step, actual=actual, expected=expected, abs_diff=abs(actual - expected)))
        if abs(actual - expected) > 1e-15:
            raise RuntimeError('POLY100_SCHEDULER_CONTRACT_FAILED')
    return rows


def synthetic_update(torch, trainer, values, arm):
    config, model, criterion, optimizer, scaler = values[:5]
    from candidate_arms import PREFIX
    model.train()
    watched = {name: parameter for name, parameter in model.named_parameters() if name.startswith(PREFIX[arm])}
    before = {name: parameter.detach().clone() for name, parameter in watched.items()}
    image = torch.rand(2, 4, 512, 640, generator=torch.Generator().manual_seed(20260924)).cuda()
    labels = torch.randint(0, 3, (2, 512, 640), generator=torch.Generator().manual_seed(20260925)).cuda()
    labels[:, :2, :3] = 255
    edges = torch.zeros(2, 512, 640, device='cuda')
    flags = torch.ones(2, device='cuda', dtype=torch.bool)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        outputs = model(image)
        losses, _, _, _ = criterion.get_loss(
            outputs,
            labels,
            edges,
            fire_folder_flags=flags,
            dense_supervision_flags=flags,
        )
        loss = losses.mean()
    if not torch.isfinite(loss):
        raise RuntimeError('SYNTHETIC_NONFINITE_LOSS')
    scaler.scale(loss).backward()
    trainer.apply_configured_gradient_clipping(model, optimizer, scaler, config)
    gradients = {name: bool(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.ne(0).any()) for name, parameter in watched.items()}
    scaler.step(optimizer)
    scaler.update()
    changed = {name: bool((parameter.detach() != before[name]).any()) for name, parameter in watched.items()}
    if arm == 'S7' and (not watched or not all(gradients.values()) or not all(changed.values())):
        raise RuntimeError('S7_PARAMETER_UPDATE_CONTRACT_FAILED')
    if arm == 'B0' and not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError('B0_GRADIENT_CONTRACT_FAILED')
    return dict(
        arm=arm,
        loss=float(loss.detach().cpu()),
        watched_parameters=len(watched),
        watched_nonzero_gradient=gradients,
        watched_updated=changed,
        scaler=scaler.state_dict(),
        output_shapes=[list(output.shape) for output in outputs],
    )


def save_restore_contract(torch, trainer, factory, rt, source):
    values = prepare(rt, trainer, factory, 'S7', 203, verify_expected=False)
    config, model, criterion, optimizer, scaler, train_generator, validation_generator = values[:7]
    synthetic_update(torch, trainer, values, 'S7')
    payload = dict(
        protocol='PREDEPLOYMENT_SAVE_RESTORE_CONTRACT',
        arm='S7',
        seed=203,
        epoch=1,
        source_manifest_sha256=source,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        rng=capture_rng(torch, train_generator, validation_generator),
    )
    path = ROOT / 'audit/recovery_contract/predeployment_checkpoint.pth'
    digest = save_torch_verified(torch, path, payload)
    size = path.stat().st_size
    restored = torch.load(path, map_location='cpu', weights_only=True)
    values2 = prepare(rt, trainer, factory, 'S7', 203, verify_expected=False)
    model2, optimizer2, scaler2 = values2[1], values2[3], values2[4]
    train_generator2, validation_generator2 = values2[5], values2[6]
    model2.load_state_dict(restored['model_state_dict'], strict=True)
    optimizer2.load_state_dict(restored['optimizer_state_dict'])
    scaler2.load_state_dict(restored['scaler_state_dict'])
    restore_rng(torch, restored['rng'], train_generator2, validation_generator2)
    passed = (
        tensor_state_sha(model2.state_dict()) == tensor_state_sha(restored['model_state_dict']) and
        bool(optimizer2.state) and
        scaler2.state_dict() == restored['scaler_state_dict'] and
        torch.equal(train_generator2.get_state(), restored['rng']['train_generator']) and
        torch.equal(validation_generator2.get_state(), restored['rng']['validation_generator'])
    )
    if not passed:
        raise RuntimeError('SAVE_RESTORE_CONTRACT_FAILED')
    path.unlink()
    return dict(status='PASS', sha256=digest, bytes=size, file_removed_after_verification=True)


def main():
    source = package_identity()
    source, guard, pref, rt, trainer, factory = setup('predeployment')
    import torch
    from engineering_checks import synthetic_loss_checks
    from candidate_arms import COUNTS, apply_arm

    torch.set_num_threads(4)
    expected = {'seeds': {}}
    for seed in SEEDS:
        expected['seeds'][str(seed)] = {'arms': {}}
        for arm in ARMS:
            values = prepare(rt, trainer, factory, arm, seed, verify_expected=False)
            initial = values[-1]
            expected['seeds'][str(seed)]['arms'][arm] = initial
            expected['seeds'][str(seed)]['baseline_state_sha256'] = initial['baseline_state_sha256']
            expected['seeds'][str(seed)]['stem_sha256'] = initial['stem_sha256']
            if sum(parameter.numel() for parameter in values[1].parameters()) != COUNTS[arm][0]:
                raise RuntimeError('PARAMETER_COUNT_MISMATCH: ' + arm)
            del values
            torch.cuda.empty_cache()
        b0 = expected['seeds'][str(seed)]['arms']['B0']
        s7 = expected['seeds'][str(seed)]['arms']['S7']
        if b0['baseline_state_sha256'] != s7['baseline_state_sha256'] or b0['stem_sha256'] != s7['stem_sha256']:
            raise RuntimeError('B0_S7_COMMON_INITIALIZATION_MISMATCH')
    write(ROOT / 'INITIALIZATION_CHECK/EXPECTED_INITIALIZATION.json', expected)

    direct = direct_stagea_module()
    config = read(config_path('B0', 203))
    rt.seed_everything(203)
    baseline = rt.build_model(config, augment=True)
    if rt.load_pretrained_if_available(baseline, config) != 301:
        raise RuntimeError('DIRECT_EQUIVALENCE_PRETRAINED_MISMATCH')
    wrapped = apply_arm(baseline, 'S7', 203).eval()
    direct_model = direct.apply_arm(baseline, 'S7', 203).eval()
    state_exact = set(wrapped.state_dict()) == set(direct_model.state_dict()) and all(torch.equal(wrapped.state_dict()[key], direct_model.state_dict()[key]) for key in wrapped.state_dict())
    probe = torch.rand(1, 4, 512, 640, generator=torch.Generator().manual_seed(20260924))
    with torch.inference_mode():
        wrapped_output = wrapped(probe)
        direct_output = direct_model(probe)
    output_max_abs = max(float((left - right).abs().max()) for left, right in zip(wrapped_output, direct_output))
    if not state_exact or output_max_abs != 0:
        raise RuntimeError('PACKAGED_S7_DIFFERS_FROM_STAGEA_EXECUTED_IMPLEMENTATION')

    loss_contract = synthetic_loss_checks(rt, factory, config, 'baseline')
    update_rows = []
    for arm in ARMS:
        values = prepare(rt, trainer, factory, arm, 203, verify_expected=True)
        update_rows.append(synthetic_update(torch, trainer, values, arm))
        del values
        torch.cuda.empty_cache()
    schedule_rows = scheduler_contract(torch, trainer)
    restore_row = save_restore_contract(torch, trainer, factory, rt, source)
    result = dict(
        status='PASS',
        training_started=False,
        data_driven_engineering_epochs=0,
        source_manifest_sha256=source,
        stagea_candidate_sha256=STAGEA_CANDIDATE_SHA,
        packaged_s7_state_exact=True,
        packaged_s7_output_max_abs=output_max_abs,
        initialization=expected,
        synthetic_loss=loss_contract,
        synthetic_updates=update_rows,
        scheduler=schedule_rows,
        save_restore=restore_row,
        audit=guard.snapshot(),
    )
    write(ROOT / 'audit/PREDEPLOYMENT_SYNTHETIC.json', result)
    print('PREDEPLOYMENT PASS: exact Stage A S7, initialization, loss, gradient, poly100 and recovery contracts', flush=True)


if __name__ == '__main__':
    main()