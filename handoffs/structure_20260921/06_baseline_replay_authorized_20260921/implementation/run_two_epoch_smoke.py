"""Approved two-epoch seed200 engineering only; never invokes validation or stage2."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.utils.data import DataLoader

from engineering_checks import load_baseline, make_criterion, runtime, sha, write_json
from structure_arms import ARMS, apply_arm, deployment_copy
from training_adapters import ThermalLoader


def assert_equal_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:
            assert_equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert_equal_tree(x,y)
    else:
        assert a==b


class CountedLoader:
    def __init__(self,loader):
        self.loader=loader
        self.batches=0
        self.items=0
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            self.batches+=1
            self.items+=len(batch[0])
            yield batch
            if self.batches%10==0 or self.batches==len(self.loader):
                print(f'ENGINEERING BATCH {self.batches}/{len(self.loader)}',flush=True)


def install_data_guard(csv_path, rows, bundle):
    allowed={str(csv_path.resolve()).lower()}
    for row in rows:
        for field in ('corrected_rgb_path','raw_thermal_path','temperature_mask_path'):
            p=Path(row[field])
            p=p if p.is_absolute() else bundle/p
            p=p.resolve()
            if 'test' in str(p).lower() or 'predict' in str(p).lower():
                raise RuntimeError('Forbidden test/prediction entry in training manifest.')
            allowed.add(str(p).lower())
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0]))
        if path.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.csv'):
            if str(path.resolve()).lower() not in allowed:
                raise RuntimeError(f'Non-allowlisted image/CSV access: {path}')
    sys.addaudithook(audit)
    return allowed


def guard_worker(_worker_id):
    dataset=torch.utils.data.get_worker_info().dataset
    install_data_guard(dataset.csv_path,dataset.rows,dataset.root)


def checkpoint_save_and_restore(model,optimizer,scaler,train_generator,path,epoch,baseline,arm,config,trainer):
    ns=np.random.get_state()
    payload=dict(epoch=epoch,engineering_only=True,stage2_resume_forbidden=True,
                 model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                 scaler_state_dict=scaler.state_dict(),train_generator_state=train_generator.get_state(),
                 python_random_state=random.getstate(),numpy_rng_name=ns[0],
                 numpy_rng_keys=torch.tensor(ns[1].astype(np.int64)),numpy_rng_pos=ns[2],
                 numpy_rng_has_gauss=ns[3],numpy_rng_cached_gaussian=ns[4],
                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(payload,path)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    restored=apply_arm(baseline,arm).cuda()
    restored.load_state_dict(saved['model_state_dict'],strict=True)
    opt=torch.optim.SGD(restored.parameters(),lr=config['LR'],momentum=config['MOMENTUM'],weight_decay=config['WD'])
    opt.load_state_dict(saved['optimizer_state_dict'])
    amp=trainer.create_cuda_grad_scaler(enabled=True,init_scale=config['AMP_INIT_SCALE'])
    amp.load_state_dict(saved['scaler_state_dict'])
    assert_equal_tree(model.state_dict(),restored.state_dict())
    assert_equal_tree(optimizer.state_dict(),opt.state_dict())
    assert_equal_tree(scaler.state_dict(),amp.state_dict())
    if not saved['optimizer_state_dict']['state']:
        raise AssertionError('No populated optimizer state to test.')
    train_generator.set_state(saved['train_generator_state'])
    random.setstate(saved['python_random_state'])
    np.random.set_state((saved['numpy_rng_name'],saved['numpy_rng_keys'].numpy().astype(np.uint32),
                        saved['numpy_rng_pos'],saved['numpy_rng_has_gauss'],saved['numpy_rng_cached_gaussian']))
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return restored,opt,amp,dict(epoch=epoch,model_exact=True,momentum_exact=True,scaler_exact=True,
                               rng_restored=True,sha256=sha(path),safe_load='weights_only')


def efficiency(model,arm,shared_desktop=False):
    from thop import profile
    deploy=deployment_copy(model,arm).cuda().eval()
    image=torch.rand(1,4,512,640,device='cuda')
    parameters=sum(p.numel() for p in deploy.parameters())
    with torch.inference_mode():
        macs,_=profile(deploy,inputs=(image,),verbose=False)
    def forward():
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            output=deploy(image)
        assert output.shape==(1,3,64,80)
    for _ in range(100):
        forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times=[]
    trials=[]
    for _ in range(10):
        values=[]
        for _ in range(200):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            forward()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        times.extend(values)
        trials.append(statistics.fmean(values))
    p95=float(np.percentile(times,95))
    return dict(parameters=parameters,macs=int(macs),gmacs=macs/1e9,flops_two_per_mac=2*macs,
                mean_ms=statistics.fmean(times),p95_ms=p95,warmup=100,trials=10,iterations_per_trial=200,
                per_trial_mean_ms=trials,amp=True,batch=1,input_shape=[1,4,512,640],
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                observed_within_budget=(macs<=7.47e9*1.15 and p95<=10),
                within_budget=None if shared_desktop else (macs<=7.47e9*1.15 and p95<=10),
                shared_desktop=shared_desktop,formal_latency_eligible=not shared_desktop,
                note='Synthetic-input stage1 architecture timing, not accuracy evaluation; engine/OS versions recorded separately.')


def run(args):
    if args.epochs!=2 or args.seed!=200:
        raise RuntimeError('This executable is restricted to seed200, exactly two engineering epochs.')
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('RTX4090 required; no CPU fallback for AMP gates.')
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite smoke output.')
    out.mkdir(parents=True,exist_ok=True)
    gates=json.loads((args.synthetic_checks/'SUMMARY.json').read_text(encoding='utf-8'))
    eligible={r['arm'] for r in gates['results'] if r.get('eligible_for_two_epoch_smoke')}
    if not set(args.arms).issubset(eligible):
        raise RuntimeError('An arm without passed CUDA synthetic gates was requested.')
    rt,trainer,factory=runtime(args.project_root)
    baseline,config,identity=load_baseline(rt,args.project_root,args.pretrained,'cuda:0')
    config.update(ROOTDATASET=str(args.bundle_root.resolve()),EPOCHS=2,SEED=200)
    frozen=rt.load_config(args.project_root/'configs/flame3/pidnet_s_fusion_manual_smoke_v11_30e.yaml')
    expected_config=copy.deepcopy(frozen)
    common_overrides=dict(ROOTDATASET=str(args.bundle_root.resolve()),EPOCHS=2,SEED=200,
                          DEVICE='cuda:0',PRETRAINED=str(args.pretrained.resolve()))
    expected_config.update(common_overrides)
    if config!=expected_config:
        raise RuntimeError('Undeclared effective config difference.')
    write_json(out/'CONFIG_DIFF_AUDIT.json',dict(common_overrides=common_overrides,
        differences={k:dict(frozen=frozen.get(k),effective=config[k]) for k in config if frozen.get(k)!=config[k]},
        undeclared_differences=[],arm_changes_external_to_yaml=True))
    assert config['BATCHSIZE']==8 and config['CROP_SIZE']==[512,640] and config['NUM_WORKERS']==4
    assert config['LR_TOTAL_EPOCHS']==100 and config['GRADIENT_CLIP_MAX_NORM']==5
    expected='project_support/artifacts/flame3_manual_smoke_v2/paired_splits/manual_smoke_v1/portable/train.csv'
    if config['TRAINSET'].replace('\\','/')!=expected:
        raise RuntimeError('Training split path drift; only frozen train.csv allowed.')
    csv_path=(args.bundle_root/expected).resolve()
    with csv_path.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=493 or len({r['sample_key'] for r in rows})!=493:
        raise RuntimeError('Frozen train493 identity/count mismatch.')
    allowed=install_data_guard(csv_path,rows,args.bundle_root.resolve())
    data_hashes={p:sha(p) for p in sorted(allowed)}
    write_json(out/'TRAIN_INPUT_SHA256_BEFORE.json',data_hashes)
    write_json(out/'EFFECTIVE_COMMON_CONFIG.json',config)
    write_json(out/'ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),identity=identity,
               test107_read=False,validation_run=False,stage2_authorized=False,shared_desktop=args.shared_desktop,
               cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
    summary=[]
    for arm in args.arms:
        dest=out/arm
        dest.mkdir()
        record=dict(arm=arm,seed=200,requested_epochs=2,completed_epochs=0,status='RUNNING',test107_read=False)
        write_json(out/'STATUS.json',record)
        print(f'SMOKE START {arm}',flush=True)
        try:
            rt.seed_everything(200)
            dataset=rt.build_dataset(config,'train')
            generator=torch.Generator().manual_seed(200)
            loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=4,pin_memory=True,
                              drop_last=True,generator=generator,worker_init_fn=guard_worker)
            model=apply_arm(baseline,arm).cuda()
            criterion=make_criterion(rt,factory,config,arm)
            optimizer=torch.optim.SGD(model.parameters(),lr=config['LR'],momentum=config['MOMENTUM'],weight_decay=config['WD'])
            scaler=trainer.create_cuda_grad_scaler(enabled=True,init_scale=config['AMP_INIT_SCALE'])
            record.update(epochs=[],checkpoints=[],declared_scientific_diff={'STAGE1_ARM':arm},train_batches_expected=len(loader))
            for epoch in range(2):
                wrapped=ThermalLoader(loader,200,epoch) if arm in ('R1','R3') else loader
                counted=CountedLoader(wrapped)
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                # Reuse the frozen optimization/AMP/clip implementation, not its
                # CLI (which would instantiate validation and select best_S).
                metrics,loss,components,_,_=trainer.run_training_epoch(model,counted,criterion,optimizer,scaler,
                                                        config,epoch,100,None,True)
                if counted.batches!=len(loader) or counted.items!=len(loader)*8 or not math.isfinite(loss):
                    raise AssertionError('Incomplete engineering epoch or non-finite loss.')
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise AssertionError('Non-finite model weights after epoch.')
                row=dict(epoch=epoch+1,batches=counted.batches,items=counted.items,loss=loss,
                         components=components,elapsed_seconds=time.perf_counter()-start,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                         fire_metrics_used_for_decision=False)
                if isinstance(wrapped,ThermalLoader):
                    row.update(thermal_degradation_counts=wrapped.counts,thermal_trace_sha256=wrapped.trace.hexdigest())
                record['epochs'].append(row)
                model,optimizer,scaler,restored=checkpoint_save_and_restore(model,optimizer,scaler,generator,
                                      dest/f'epoch{epoch+1}_engineering_only.pth',epoch+1,baseline,arm,config,trainer)
                record['checkpoints'].append(restored)
                record['completed_epochs']=epoch+1
                write_json(dest/'RESULT.json',record)
                write_json(out/'STATUS.json',record)
            del optimizer,scaler,criterion,loader,dataset,counted,wrapped
            model.cpu()
            torch.cuda.empty_cache()
            record['efficiency']=efficiency(model,arm,args.shared_desktop)
            record['status']='TWO_EPOCH_ENGINEERING_PASS'
            if args.shared_desktop:
                record['budget_status']='LATENCY_PENDING_UNSHARED_MEASUREMENT'
            elif not record['efficiency']['within_budget']:
                record['budget_status']='OVER_BUDGET_NOT_DESIGN_LAYER_ELIGIBLE'
            else:
                record['budget_status']='WITHIN_BUDGET'
            del model
        except Exception as exc:
            record.update(status='ENGINEERING_FAILED_STOP_ARM',error=repr(exc),traceback=traceback.format_exc())
        summary.append(record)
        write_json(dest/'RESULT.json',record)
        write_json(out/'SUMMARY.json',dict(results=summary,stage2_authorized=False,test107_read=False))
        print(f"SMOKE END {arm}: {record['status']}",flush=True)
        torch.cuda.empty_cache()
    after={p:sha(p) for p in sorted(allowed)}
    write_json(out/'TRAIN_INPUT_SHA256_AFTER.json',after)
    if after!=data_hashes:
        raise RuntimeError('Training source data changed; stop and report.')
    write_json(out/'STATUS.json',dict(status='COMPLETE_ENGINEERING_ONLY',results=[{k:r[k] for k in ('arm','status','completed_epochs')} for r in summary],
                                     train_input_hashes_unchanged=True,stage2_authorized=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--synthetic-checks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=ARMS,required=True)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--seed',type=int,default=200)
    p.add_argument('--shared-desktop',action='store_true',help='Timing is diagnostic only, not a formal latency verdict.')
    run(p.parse_args())
