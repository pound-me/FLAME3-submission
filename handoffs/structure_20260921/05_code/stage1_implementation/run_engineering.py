"""Stage-1 synthetic gate runner. No train/val/test dataset is built here."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

import torch

from engineering_checks import (ARMS, augmentation_checks, checkpoint_roundtrip, load_baseline,
                                model_checks, runtime, sha, synthetic_loss_checks, write_json)
from structure_arms import apply_arm, deployment_copy


def measure(candidate,arm,device):
    from thop import profile
    model = deployment_copy(candidate,arm).to(device).eval()
    image = torch.rand(1,4,512,640,device=device)
    with torch.inference_mode():
        macs,thop_parameters = profile(model,inputs=(image,),verbose=False)
    return dict(parameters=sum(p.numel() for p in model.parameters()),thop_parameters=int(thop_parameters),
                macs=int(macs),gmacs=float(macs/1e9),flops_2mac=int(2*macs),
                under_gmac_budget=macs<=7.47e9*1.15,
                convention='THOP frozen baseline convention; elementwise gate/reductions may be uncounted')


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing gate run.')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cpu' if args.phase=='cpu' else 'cuda:0')
    if device.type=='cuda' and (not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('CUDA engineering gates require the approved RTX4090.')
    rt,trainer,factory = runtime(args.project_root)
    baseline,config,identity = load_baseline(rt,args.project_root,args.pretrained,str(device))
    env = dict(python=sys.version,torch=torch.__version__,device=str(device),
               gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
               phase=args.phase,identity=identity,training_epochs=0,dataset_files_read=0,test107_read=False)
    sources = [args.project_root/'src/baseline_runtime.py',args.project_root/'src/train_baseline_v11.py',
               args.project_root/'src/custom_losses.py',args.project_root/'src/flame3_dataset.py',
               args.project_root/'third_party/RoboFireFuseNet/models/pidnet.py',
               args.project_root/'third_party/RoboFireFuseNet/models/pidnet_utils.py',
               args.project_root/'third_party/RoboFireFuseNet/utils/total_loss.py',
               args.project_root/'configs/flame3/pidnet_s_fusion_manual_smoke_v11_30e.yaml',args.pretrained]
    before = {str(p.resolve()):sha(p) for p in sources}
    write_json(output/'INPUT_SHA256_BEFORE.json',before)
    write_json(output/'ENVIRONMENT.json',env)
    results = []
    for arm in args.arms:
        print(f'START {args.phase} {arm}',flush=True)
        record = dict(arm=arm,phase=args.phase,engineering_complete=False,two_epoch_smoke='NOT_RUN')
        try:
            candidate,fp32 = model_checks(baseline,arm,device,amp=False)
            record['fp32'] = fp32
            record['loss_checks'] = synthetic_loss_checks(rt,factory,config,arm)
            if arm in ('R1','R3'):
                record['augmentation'] = augmentation_checks()
            record['checkpoint'] = checkpoint_roundtrip(candidate,arm,output,config,trainer)
            record['complexity'] = measure(candidate,arm,device)
            if device.type=='cuda':
                _,amp = model_checks(baseline,arm,device,amp=True)
                record['amp'] = amp
                numeric = fp32['numeric_pass'] and amp['numeric_pass']
            else:
                numeric = fp32['numeric_pass']
            record['status'] = 'SYNTHETIC_GATES_PASS' if numeric else 'NUMERIC_GATE_FAILED_STOP_ARM'
            record['eligible_for_two_epoch_smoke'] = numeric and device.type=='cuda'
        except Exception as exc:
            record.update(status='ENGINEERING_ERROR_STOP_ARM',eligible_for_two_epoch_smoke=False,
                          error=repr(exc),traceback=traceback.format_exc())
        results.append(record)
        write_json(output/f'{arm}.json',record)
        write_json(output/'SUMMARY.json',dict(environment=env,results=results,stage2_authorized=False))
        print(f"END {args.phase} {arm}: {record['status']}",flush=True)
        if device.type=='cuda':
            torch.cuda.empty_cache()
    after = {str(p.resolve()):sha(p) for p in sources}
    write_json(output/'INPUT_SHA256_AFTER.json',after)
    if before!=after:
        raise RuntimeError('Frozen source or pretrained input changed during checks.')
    print(json.dumps({r['arm']:r['status'] for r in results},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['cpu','cuda'],required=True)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    run(p.parse_args())
