"""Frozen stage2 policy and file access boundary, independent of Torch."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ARMS=('R2','R3')
SEEDS=(200,201,202)
CONFIG_REL='configs/flame3/pidnet_s_fusion_manual_smoke_v11_30e.yaml'
TRAIN_REL='project_support/artifacts/flame3_manual_smoke_v2/paired_splits/manual_smoke_v1/portable/train.csv'
VAL_REL='project_support/artifacts/flame3_manual_smoke_v2/paired_splits/manual_smoke_v1/portable/val.csv'
ANNOTATION_REL='project_support/data/flame3_manual_smoke_v2_annotation'
PROTOCOL_ID='flame3_structure_stage2_r2r3_append_20260918_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def effective_config(frozen,bundle,seed):
    if seed not in SEEDS:
        raise ValueError('Unapproved seed')
    config=copy.deepcopy(frozen)
    changes=dict(ROOTDATASET=str(bundle.resolve()),PRETRAINED=str((bundle/'weights/PIDNet_S_ImageNet.pth.tar').resolve()),
                 FLAME3_MANUAL_SMOKE_V2_ANNOTATION_PACKAGE=str((bundle/ANNOTATION_REL).resolve()),
                 DEVICE='cuda:0',SEED=seed,EPOCHS=30)
    config.update(changes)
    assert config['TRAINSET'].replace('\\','/')==TRAIN_REL
    assert config['VALIDSET'].replace('\\','/')==VAL_REL
    assert config['BATCHSIZE']==8 and config['NUM_WORKERS']==4 and config['CROP_SIZE']==[512,640]
    assert config['LR_TOTAL_EPOCHS']==100 and config['GRADIENT_CLIP_MAX_NORM']==5
    assert config['PRETRAIN_SKIP_KEYS']==['conv1.0.weight']
    assert not config['FLAME3_TARGETED_SAMPLING_ENABLED'] and config['TRAINING_OBJECTIVE']=='partial_label'
    differences={k:dict(before=frozen.get(k),after=config[k]) for k in config if frozen.get(k)!=config[k]}
    assert set(differences).issubset(changes)
    return config,dict(differences=differences,undeclared_differences=[],
                       selection_score_S_use='RECORD_ONLY',selection_applied=False)


def normalized(path):
    return str(Path(path).resolve()).casefold()


def input_paths(csv_path,rows,bundle):
    allowed={normalized(csv_path)}
    for row in rows:
        for field in ('corrected_rgb_path','raw_thermal_path','temperature_mask_path'):
            p=Path(row[field])
            p=p if p.is_absolute() else bundle/p
            text=normalized(p)
            if 'test' in text or 'predict' in text:
                raise RuntimeError('Forbidden train/val entry: '+text)
            allowed.add(text)
    return allowed


def validate_image_access(path,allowed):
    p=Path(os.fsdecode(path))
    text=normalized(p)
    if ('test107' in text or 'test_blind' in text) and p.suffix.lower() not in ('.py','.pyc'):
        raise RuntimeError('Sealed test107 access: '+text)
    if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.csv') and text not in allowed:
        raise RuntimeError('Non-allowlisted data access: '+text)


def install_guard(allowed):
    allowed={normalized(p) for p in allowed}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            validate_image_access(args[0],allowed)
    sys.addaudithook(audit)


def verify_package(root):
    manifest=read(root/'MANIFEST.json')
    for row in manifest['files']:
        if sha(root/row['relative'])!=row['sha256']:
            raise RuntimeError('Source package drift: '+row['relative'])
    return sha(root/'MANIFEST.json')


def assert_finite(value):
    if isinstance(value,dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value,(list,tuple)):
        for item in value:
            assert_finite(item)
    elif isinstance(value,float) and not math.isfinite(value):
        raise RuntimeError('Non-finite metric or loss')


def descriptive_summary(records):
    assert [r['epoch'] for r in records]==list(range(1,31))
    selected=[r for r in records if 26<=r['epoch']<=30]
    names=('smoke_iou_A001_A047','background_iou_A001_A047','three_class_miou_A001_A047',
           'no_fire_joint_false_positive_ratio','selection_score_S','fire_heat_iou_full134')
    window={k:sum(r['validation'][k] for r in selected)/5 for k in names}
    best=max(records,key=lambda r:r['validation']['selection_score_S'])
    return dict(window_epochs=[26,27,28,29,30],window_means=window,
                best_S_record_only=dict(epoch=best['epoch'],value=best['validation']['selection_score_S']),
                passed=None,decision='PENDING_STAGE3_PAIRED_SMOKE_AND_ROBUSTNESS',
                S_used_for_decision=False,Fire_used_for_positive_decision=False)
