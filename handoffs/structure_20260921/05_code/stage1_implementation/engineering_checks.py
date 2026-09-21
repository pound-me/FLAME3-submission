"""Synthetic-only numeric gates. No dataset construction or test split access."""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from structure_arms import ARMS, apply_arm, assert_unchanged_state, deployment_copy
from training_adapters import DualBoundaryCriterion, MaskedBoundaryBCE, degrade_thermal, dual_boundary_targets

EXPECTED = {'S1': (8050407,7624323), 'S3': (7558631,7465475), 'S4': (7454215,7361059),
            'S6': (7864807,7771651), 'R1': (7717095,7623939), 'R2': (7717320,7624164),
            'R3': (7717320,7624164), 'B1': (7717095,7623939), 'E1': (7643047,7549891)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest().upper()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def runtime(project):
    sys.path.insert(0, str(Path(project).resolve()/'src'))
    import baseline_runtime
    import train_baseline_v11
    from custom_losses import build_training_criterion
    return baseline_runtime, train_baseline_v11, build_training_criterion


def load_baseline(rt, project, pretrained, device='cpu'):
    config_path = Path(project)/'configs/flame3/pidnet_s_fusion_manual_smoke_v11_30e.yaml'
    config = rt.load_config(config_path)
    config.update(SEED=200, DEVICE=device, PRETRAINED=str(Path(pretrained).resolve()))
    if config['PRETRAIN_SKIP_KEYS'] != ['conv1.0.weight']:
        raise RuntimeError('Frozen random four-channel-stem skip changed.')
    rt.seed_everything(200)
    model = rt.build_model(config, augment=True)
    loaded = rt.load_pretrained_if_available(model, config)
    if loaded <= 0 or sum(p.numel() for p in model.parameters()) != 7717095:
        raise AssertionError('Baseline architecture or pretrained initialization mismatch.')
    return model, config, dict(config_sha256=sha(config_path),pretrained_sha256=sha(pretrained),
                                matched_pretrained_tensors=loaded,seed=200)


def make_criterion(rt, factory, config, arm):
    criterion = factory(rt.TotalLoss(config), config)
    return DualBoundaryCriterion(criterion) if arm == 'B1' else criterion


def synthetic_loss_checks(rt, factory, config, arm):
    device = torch.device(config['DEVICE'])
    original = make_criterion(rt, factory, config, 'baseline')
    candidate = make_criterion(rt, factory, config, arm)
    rng = torch.Generator().manual_seed(190917)
    labels = torch.zeros(4, 16, 20, dtype=torch.long, device=device)
    labels[0, 2:7, 2:7] = 1
    labels[0, 9:13, 11:16] = 2
    labels[1, 4:9, 4:9] = 2
    labels[:, :2, :3] = 255
    labels[3] = 255
    fire = torch.tensor([True, True, False, True], device=device)
    dense = torch.tensor([True, False, False, True], device=device)
    logits = torch.randn(4, 3, 16, 20, generator=rng).to(device).requires_grad_()
    other = logits.detach().clone()
    other.permute(0,2,3,1)[labels.eq(255)] += torch.tensor([30.,-40.,50.], device=device)
    original_loss, _ = original.set_criterion(logits, labels, fire, dense_supervision_flags=dense)
    loss, _ = candidate.set_criterion(logits, labels, fire, dense_supervision_flags=dense)
    changed_ignore_loss, _ = candidate.set_criterion(other, labels, fire, dense_supervision_flags=dense)
    if not torch.equal(original_loss, loss) or not torch.equal(loss, changed_ignore_loss):
        raise AssertionError('Set loss changed for identical logits or used Ignore pixels.')
    loss.backward()
    if float(logits.grad.permute(0,2,3,1)[labels.eq(255)].abs().max()) != 0:
        raise AssertionError('Ignore has a semantic gradient.')
    if not torch.isfinite(logits.grad).all():
        raise AssertionError('Non-finite synthetic semantic gradient.')
    result = dict(identical_logits_set_loss_exact=True, ignore_loss_invariant=True, ignore_gradient_exact_zero=True)
    if arm == 'B1':
        target, valid = dual_boundary_targets(labels, fire, dense)
        if bool(valid[labels.eq(255)].any()):
            raise AssertionError('B1 Ignore was not masked.')
        if bool(valid[1, 15, 19]):
            raise AssertionError('Unknown partial non-Fire pixel became a negative boundary label.')
        if not bool(valid[2, 15, 19]):
            raise AssertionError('Known No-Fire Background was incorrectly treated as unknown.')
        target = target.masked_fill(~valid,255)
        boundary_logits = torch.randn(4,1,16,20,generator=rng).to(device).requires_grad_()
        bd = MaskedBoundaryBCE(config['BD_WEIGHT'])
        first = bd(boundary_logits,target)
        altered = boundary_logits.detach().clone()
        altered[:,0][~valid] += 80
        if not torch.equal(first,bd(altered,target)):
            raise AssertionError('B1 masked boundary values affected loss.')
        first.backward()
        if float(boundary_logits.grad[:,0][~valid].abs().max()) != 0:
            raise AssertionError('B1 Ignore has a boundary gradient.')
        empty = bd(boundary_logits,torch.full_like(target,255))
        if float(empty.detach()) != 0 or not torch.isfinite(empty):
            raise AssertionError('All-Ignore B1 loss must be finite zero.')
        result.update(boundary_ignore_loss_invariant=True,boundary_ignore_gradient_zero=True,
                      boundary_all_ignore_zero=True,partial_unknown_masked=True,known_background_retained=True)
    return result


def augmentation_checks():
    image = torch.ones(4,512,640)*.6
    rng_state = torch.get_rng_state().clone()
    records = []
    for kind in ('clean','noise','blur','dropout'):
        a,r = degrade_thermal(image,200,0,'synthetic-only',forced=kind)
        b,s = degrade_thermal(image,200,0,'synthetic-only',forced=kind)
        assert torch.equal(a,b) and r==s and torch.equal(a[:3],image[:3])
        assert torch.isfinite(a).all() and float(a.min())>=0 and float(a.max())<=1
        if kind=='dropout':
            assert int(a[3].eq(0).sum())==128*128
        records.append(r)
    assert torch.equal(rng_state,torch.get_rng_state())
    return dict(rgb_unchanged=True,deterministic_R1_R3_pairing=True,global_rng_unchanged=True,dropout_exact_pixels=16384,forced_cases=records)


def checkpoint_roundtrip(model, arm, output, config, trainer):
    optimizer = torch.optim.SGD(model.parameters(),lr=config['LR'],momentum=config['MOMENTUM'],weight_decay=config['WD'])
    scaler = trainer.create_cuda_grad_scaler(enabled=False,init_scale=config['AMP_INIT_SCALE'])
    path = Path(output)/f'{arm}_synthetic_roundtrip.pth'
    # This preliminary roundtrip has no optimizer update; the two-epoch runner also
    # verifies populated momentum, scaler and RNG states before reporting gate 9.
    torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                    scaler_state_dict=scaler.state_dict(),epoch=0,engineering_only=True),path)
    payload = torch.load(path,map_location='cpu',weights_only=True)
    restored = copy.deepcopy(model).cpu()
    restored.load_state_dict(payload['model_state_dict'],strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    scaler.load_state_dict(payload['scaler_state_dict'])
    for k,v in restored.state_dict().items():
        if not torch.equal(v,model.state_dict()[k].detach().cpu()):
            raise AssertionError('Checkpoint roundtrip tensor mismatch.')
    return dict(state_roundtrip=True,weights_only=True,sha256=sha(path),path=str(path),
                populated_optimizer_resume='PENDING_TWO_EPOCH_SMOKE')


def model_checks(baseline, arm, device, amp=False):
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []
    candidate = apply_arm(baseline,arm)
    if not torch.equal(cpu_rng,torch.get_rng_state()):
        raise AssertionError('Arm construction changed global CPU RNG.')
    if any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if cuda_rng else [])):
        raise AssertionError('Arm construction changed global CUDA RNG.')
    unchanged = assert_unchanged_state(baseline,candidate,arm)
    training_parameters = sum(p.numel() for p in candidate.parameters())
    deployed = deployment_copy(candidate,arm)
    deployment_parameters = sum(p.numel() for p in deployed.parameters())
    if (training_parameters,deployment_parameters)!=EXPECTED[arm]:
        raise AssertionError(f'Parameter mismatch: {arm}: {training_parameters}/{deployment_parameters}')
    candidate = candidate.to(device).eval()
    deployed = deployed.to(device).eval()
    reference = copy.deepcopy(baseline).to(device).eval() if arm in ('R2','R3') else None
    rng = torch.Generator().manual_seed(20260917)
    probes = [torch.rand(1,4,512,640,generator=rng),torch.zeros(1,4,512,640),torch.ones(1,4,512,640)]
    cases = []
    for index,image in enumerate(probes):
        image = image.to(device)
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            out = candidate(image)
            if [list(x.shape) for x in out] != [[1,3,64,80],[1,3,64,80],[1,1,64,80]]:
                raise AssertionError('Frozen output shapes changed.')
            if not all(bool(torch.isfinite(x).all()) for x in out):
                raise AssertionError('Non-finite forward output.')
            dep = deployed(image)
            record = dict(probe=index,shapes=[list(x.shape) for x in out],finite=True)
            if arm == 'S1':
                difference = float((out[1].float()-dep.float()).abs().max())
                record.update(fold_max_abs=difference,fold_pass=difference<1e-4,tolerance=1e-4)
            if reference is not None:
                ref = reference(image)
                difference = max(float((a.float()-b.float()).abs().max()) for a,b in zip(out,ref))
                fr,ft,g = candidate.conv1[0].components(image)
                initial_stem = candidate.conv1[0](image)
                ref_stem = reference.conv1[0](image)
                record.update(initial_max_abs_all_heads=difference,initial_pass=difference<1e-6,
                              initial_stem_max_abs=float((initial_stem.float()-ref_stem.float()).abs().max()),
                              initial_gate_min=float(g.min()),initial_gate_max=float(g.max()),tolerance=1e-6)
            cases.append(record)
    gradient = None
    if arm in ('R2','R3'):
        candidate.zero_grad(set_to_none=True)
        fr,ft,g = candidate.conv1[0].components(probes[0].to(device))
        g.mean().backward()
        final_gradient = candidate.conv1[0].gate[-1].bias.grad
        gradient = float(final_gradient.abs().max())
        if gradient<=0 or not torch.isfinite(final_gradient).all():
            raise AssertionError('2*sigmoid gate has no finite initial learning signal.')
    return candidate.cpu(),dict(unchanged=unchanged,training_parameters=training_parameters,
            deployment_parameters=deployment_parameters,precision='CUDA_AMP_FP16' if amp else 'FP32',
            cases=cases,gate_final_bias_gradient=gradient,construction_cpu_rng_preserved=True,
            construction_cuda_rng_preserved=True if device.type=='cuda' else None,
            numeric_pass=all(c.get('fold_pass',True) and c.get('initial_pass',True) for c in cases))
