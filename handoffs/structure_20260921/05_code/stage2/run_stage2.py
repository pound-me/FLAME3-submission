"""Approved six-arm stage2 worker. No test or robustness entrypoint is called."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'project_support_night_20260824/.deps'))
sys.path.insert(0,str(ROOT/'implementation'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from engineering_checks import runtime,make_criterion
from structure_arms import apply_arm,assert_unchanged_state
from training_adapters import ThermalLoader
from stage2_protocol import (ARMS,SEEDS,CONFIG_REL,TRAIN_REL,VAL_REL,ANNOTATION_REL,PROTOCOL_ID,
    sha,read,write,effective_config,normalized,input_paths,install_guard,verify_package,assert_finite,descriptive_summary)


def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def data_inventory(root):
    bundle=root.parent
    train,val=bundle/TRAIN_REL,bundle/VAL_REL
    tr,vr=load_csv(train),load_csv(val)
    if len(tr)!=493 or len(vr)!=134:
        raise RuntimeError('Train493/val134 counts changed')
    train_keys={r['sample_key'] for r in tr}
    val_keys={r['sample_key'] for r in vr}
    if len(train_keys)!=493 or len(val_keys)!=134 or train_keys&val_keys:
        raise RuntimeError('Duplicate keys or train/validation overlap')
    if sum(r['sample_class']=='No Fire' for r in vr)!=35:
        raise RuntimeError('Frozen No-Fire validation pool changed')
    allowed=input_paths(train,tr,bundle)|input_paths(val,vr,bundle)
    annotation=bundle/ANNOTATION_REL
    manifest=annotation/'flame3_threeclass_annotation_manifest_150.json'
    freeze=annotation/'final_incremental_audit_20260815/ANNOTATION_FINAL_FREEZE.json'
    frozen=read(freeze)
    if frozen['status']!='fully_frozen_for_baseline_and_manual_smoke_v1_training' or frozen['test_images_or_labels_read'] is not False:
        raise RuntimeError('Manual target freeze mismatch')
    items=[x for x in read(manifest)['items'] if x['split']=='val']
    if sorted(x['annotation_id'] for x in items)!=[f'A{i:03d}' for i in range(1,48)]:
        raise RuntimeError('Manual dev IDs changed')
    if not {x['sample_key'] for x in items}.issubset(val_keys):
        raise RuntimeError('Manual dev keys outside validation')
    allowed|={normalized(annotation/'completed_masks'/(x['annotation_id']+'.png')) for x in items}
    hashed=allowed|{normalized(manifest),normalized(freeze)}
    return allowed,hashed,dict(train=493,validation=134,manual_dev=47,no_fire=35)


def preflight(root):
    manifest_sha=verify_package(root)
    allowed,hashed,counts=data_inventory(root)
    install_guard(allowed)
    hashes={p:sha(p) for p in sorted(hashed)}
    old=read(root/'stage1_evidence/S3/TRAIN_INPUT_SHA256_BEFORE.json')
    if any(hashes.get(p.casefold())!=h for p,h in old.items()):
        raise RuntimeError('Train input changed since completed Stage1')
    gates=read(root/'stage1_evidence/cuda/SUMMARY.json')
    if {r['arm'] for r in gates['results'] if r['status']=='SYNTHETIC_GATES_PASS'}!=set(ARMS):
        raise RuntimeError('Missing approved six-arm CUDA gates')
    for arm in ARMS:
        record=read(root/'stage1_evidence'/arm/'RESULT.json')
        if record['status']!='TWO_EPOCH_ENGINEERING_PASS' or record['completed_epochs']!=2:
            raise RuntimeError('Missing two-epoch gate: '+arm)
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Approved RTX4090 is required')
    free=shutil.disk_usage(root).free
    if free<20*1024**3:
        raise RuntimeError('Less than 20GiB free for checkpoint retention')
    result=dict(protocol=PROTOCOL_ID,status='PASS',counts=counts,input_sha256=hashes,
                source_manifest_sha256=manifest_sha,gpu=torch.cuda.get_device_name(0),
                python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,free_bytes=free,
                test107_read=False,stage1_checkpoint_loaded=False,formal_latency_required_before_training=False)
    path=root/'PREFLIGHT.json'
    if path.exists():
        old_pre=read(path)
        if old_pre['input_sha256']!=hashes or old_pre['source_manifest_sha256']!=manifest_sha:
            raise RuntimeError('Preflight inputs changed')
    else:
        write(path,result)
    return result,allowed


def worker_guard(_worker_id):
    dataset=torch.utils.data.get_worker_info().dataset
    install_guard(input_paths(dataset.csv_path,dataset.rows,dataset.root))


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


def rng_state(generator):
    ns=np.random.get_state()
    return dict(train_generator=generator.get_state(),python=random.getstate(),numpy_name=ns[0],
                numpy_keys=torch.tensor(ns[1].astype(np.int64)),numpy_pos=ns[2],numpy_has_gauss=ns[3],
                numpy_cached=ns[4],torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state,generator):
    generator.set_state(state['train_generator'])
    random.setstate(state['python'])
    np.random.set_state((state['numpy_name'],state['numpy_keys'].numpy().astype(np.uint32),
                        state['numpy_pos'],state['numpy_has_gauss'],state['numpy_cached']))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path,model,optimizer,scaler,generator,config,arm,seed,epoch,source_hash,record):
    payload=dict(protocol=PROTOCOL_ID,stage=2,engineering_only=False,arm=arm,seed=seed,epoch=epoch,
                 config=config,source_manifest_sha256=source_hash,model_state_dict=model.state_dict(),
                 optimizer_state_dict=optimizer.state_dict(),scaler_state_dict=scaler.state_dict(),
                 rng=rng_state(generator),epoch_record=record,S_selection_applied=False)
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp)
    tmp.replace(path)


def verify_resume_header(saved,arm,seed,config,source_hash):
    if saved.get('engineering_only') or saved.get('stage')!=2 or saved.get('protocol')!=PROTOCOL_ID:
        raise RuntimeError('Only this formal stage2 checkpoint can be resumed')
    if (saved['arm'],saved['seed'],saved['config'],saved['source_manifest_sha256'])!=(arm,seed,config,source_hash):
        raise RuntimeError('Resume identity/config/source mismatch')


def run_training(root,arm,seed,resume):
    if arm not in ARMS or seed not in SEEDS:
        raise RuntimeError('Arm/seed not authorized')
    pref,allowed=preflight(root)
    source=root/'source'
    rt,trainer,factory=runtime(source)
    config,diff=effective_config(rt.load_config(source/CONFIG_REL),root.parent,seed)
    folder=root/'runs'/arm/f'seed{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError('Non-empty formal run; no automatic restart or overwrite')
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'CONFIG_DIFF.json',diff)
    write(folder/'resolved_config.json',config)
    rt.seed_everything(seed)
    training=rt.build_dataset(config,'train')
    validation=rt.build_dataset(config,'val')
    targets=trainer.load_frozen_manual_dev_targets(config['FLAME3_MANUAL_SMOKE_V2_ANNOTATION_PACKAGE'])
    generator=torch.Generator().manual_seed(seed)
    sampler=trainer.build_flame3_train_sampler(training,config,generator)
    if sampler is not None:
        raise RuntimeError('Unexpected non-frozen training sampler')
    train_loader=DataLoader(training,batch_size=8,shuffle=True,num_workers=4,pin_memory=True,
                            drop_last=True,generator=generator,worker_init_fn=worker_guard)
    val_loader=DataLoader(validation,batch_size=8,shuffle=False,num_workers=4,pin_memory=True,
                          worker_init_fn=worker_guard)
    baseline=rt.build_model(config,augment=True)
    matched=rt.load_pretrained_if_available(baseline,config)
    if matched!=301:
        raise RuntimeError('Unexpected pretrained initialization')
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all()
    model=apply_arm(baseline,arm,seed=seed)
    unchanged=assert_unchanged_state(baseline,model,arm)
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    del baseline
    model=model.cuda()
    criterion=make_criterion(rt,factory,config,arm)
    optimizer=torch.optim.SGD(model.parameters(),lr=config['LR'],momentum=config['MOMENTUM'],weight_decay=config['WD'])
    scaler=trainer.create_cuda_grad_scaler(enabled=True,init_scale=config['AMP_INIT_SCALE'])
    record_file=folder/'metrics.jsonl'
    records=[]
    start_epoch=0
    if resume:
        saved=torch.load(folder/'last.pth',map_location='cpu',weights_only=True)
        verify_resume_header(saved,arm,seed,config,pref['source_manifest_sha256'])
        model.load_state_dict(saved['model_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict'])
        restore_rng(saved['rng'],generator)
        start_epoch=int(saved['epoch'])
        records=[read(folder/'epochs'/f'epoch{i:02d}.json') for i in range(1,start_epoch+1)]
        if records[-1]!=saved['epoch_record']:
            raise RuntimeError('Checkpoint/epoch journal mismatch; stop for audit')
        del saved
    write(folder/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
          cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),seed=seed,
          matched_pretrained=matched,source_manifest_sha256=pref['source_manifest_sha256'],unchanged=unchanged,
          cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
          cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
          stage1_checkpoint_loaded=False,test107_read=False,desktop_apps_started=False))
    record_file.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    for epoch in range(start_epoch,30):
        write(folder/'STATUS.json',dict(status='TRAINING',arm=arm,seed=seed,current_epoch=epoch+1,completed_epochs=epoch,
                                       started_unix=time.time(),test107_read=False))
        print(f'STAGE2 {arm} seed={seed} epoch={epoch+1}/30',flush=True)
        began=time.perf_counter()
        wrapped=ThermalLoader(train_loader,seed,epoch) if arm=='R1' else train_loader
        counted=CountedLoader(wrapped)
        torch.cuda.reset_peak_memory_stats()
        train_metrics,loss,components,_,_=trainer.run_training_epoch(model,counted,criterion,optimizer,scaler,config,epoch,100,None,True)
        if (counted.batches,counted.items)!=(61,488):
            raise RuntimeError('Incomplete training epoch')
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError('Non-finite model parameter')
        checked_val=CountedLoader(val_loader)
        val,val_loss,val_components,_=trainer.run_validation(model,checked_val,criterion,config,epoch,None,True,targets)
        if (checked_val.batches,checked_val.items)!=(17,134):
            raise RuntimeError('Incomplete validation epoch')
        classes=val['manual_smoke_v2']['manual_A001_A047']['classes']
        val['background_iou_A001_A047']=float(classes['background']['iou'])
        val['three_class_miou_A001_A047']=sum(float(x['iou']) for x in classes.values())/3
        record=dict(epoch=epoch+1,arm=arm,seed=seed,train_loss=loss,validation_loss=val_loss,
                    train=train_metrics,validation=val,loss_components=dict(train=components,validation=val_components),
                    train_batches=counted.batches,train_items=counted.items,validation_items=checked_val.items,
                    epoch_elapsed_seconds=time.perf_counter()-began,lr_total_epochs=100,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                    S_use='RECORD_ONLY',checkpoint_selection='NONE_FIXED_WINDOW_26_30',test107_read=False)
        if arm=='R1':
            record['thermal_degradation_counts']=wrapped.counts
            record['thermal_trace_sha256']=wrapped.trace.hexdigest()
        assert_finite(record)
        # Commit an epoch journal before last.pth; resume rejects inconsistent tails.
        write(folder/'epochs'/f'epoch{epoch+1:02d}.json',record)
        save_checkpoint(folder/'last.pth',model,optimizer,scaler,generator,config,arm,seed,epoch+1,
                        pref['source_manifest_sha256'],record)
        if epoch+1>=26:
            shutil.copy2(folder/'last.pth',folder/f'epoch{epoch+1:02d}.pth')
        with record_file.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        records.append(record)
        write(folder/'STATUS.json',dict(status='EPOCH_COMPLETE',arm=arm,seed=seed,completed_epochs=epoch+1,
                                       smoke_iou=val['smoke_iou_A001_A047'],S_record_only=val['selection_score_S'],
                                       updated_unix=time.time(),test107_read=False))
        print(f"EPOCH_COMPLETE {arm} seed={seed} epoch={epoch+1} Smoke={val['smoke_iou_A001_A047']:.6f} S(record)={val['selection_score_S']:.6f}",flush=True)
    after={p:sha(p) for p in pref['input_sha256']}
    if after!=pref['input_sha256']:
        raise RuntimeError('Read-only input changed during training')
    verify_package(root)
    hashes={p.name:sha(p) for p in folder.glob('*.pth')}
    write(folder/'CHECKPOINT_SHA256.json',hashes)
    write(folder/'RESULT.json',dict(status='COMPLETE_30_EPOCHS',arm=arm,seed=seed,epochs=30,
          descriptive=descriptive_summary(records),input_hashes_unchanged=True,test107_read=False,
          checkpoint_sha256=hashes,stage3_automatically_started=False))
    write(folder/'STATUS.json',dict(status='COMPLETE_30_EPOCHS',arm=arm,seed=seed,completed_epochs=30,test107_read=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.preflight_only:
        pref,_=preflight(ROOT)
        print(json.dumps({k:v for k,v in pref.items() if k!='input_sha256'},indent=2),flush=True)
        return
    if args.arm is None or args.seed is None:
        parser.error('arm and seed required')
    try:
        run_training(ROOT,args.arm,args.seed,args.resume)
    except BaseException as exc:
        write(ROOT/'runs'/args.arm/f'seed{args.seed}'/'FAILURE.json',dict(status='FAILED_STOP_ARM',
              error=repr(exc),traceback=traceback.format_exc(),test107_read=False))
        raise


if __name__=='__main__':
    main()
