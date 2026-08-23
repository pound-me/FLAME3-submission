from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

EXPECTED_SEEDS=(200,201,202)
EXPECTED_TEST_CSV_SHA256="3ebdccd489d4e71ce87c0bd6f7b75faf34cc051b89d1da137da1ac0f6b7b9355"
EXPECTED_STRATA_SHA256="0e6dd36bb50b81dd6de47c17a9b8041f7218f6c450e47c3503d6684ad7ff71ed"
EXPECTED_PREREG_SHA256="9312759b787c5526cad63469037960ea80a0085cfbe090965d8660c9f173ea43"
EXPECTED_FROZEN_RESULT_SHA256="81b0c5f83328d7b646b7ca2ed74c5541e02ca6e49fcc493efee07bb22432e260"
EXPECTED_CHECKPOINT_SHA256={
("baseline",200):"1770dd6d94fe9bcee378c8a3674ad8dbd7feeb99b710befee6eedd060957f206",
("baseline",201):"79bf3911aba25c5d484945735b9497ebbec3813feb174efb1f47e25543356a8e",
("baseline",202):"b78e549e154316c404600d955b2f23af058397669fae61ddb95be33a6d2dd97f",
("v11",200):"97e9d5902da9eb8beaa240760c41aa5e68086709ced6f8cb3f6481197a4cec6f",
("v11",201):"32f7e5df88556d3efbee7227e31bfac19677691002a67c24a68e116474c799e7",
("v11",202):"7e92ae0686b279c9f24f11d02bd8134e62454a10d489d079eb1d86262ca022cf",
}
CLASS_NAMES=("background","smoke","fire_heat")
METRIC_DIRECTIONS={"background_iou":1,"smoke_iou":1,"fire_heat_iou":1,"three_class_miou":1,"equal_fire_smoke_S":1,"no_fire_joint_fp":-1}


def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument("--project-root",type=Path,required=True); p.add_argument("--bundle-root",type=Path,required=True)
 p.add_argument("--test-csv",type=Path,required=True); p.add_argument("--strata-manifest",type=Path,required=True)
 p.add_argument("--preregistration",type=Path,required=True); p.add_argument("--frozen-result",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda:0")
 p.add_argument("--batch-size",type=int,default=4); p.add_argument("--num-workers",type=int,default=0)
 p.add_argument("--amp",action="store_true"); p.add_argument("--preflight-only",action="store_true")
 return p.parse_args()


def sha256_file(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): d.update(b)
 return d.hexdigest()


def read_csv(path):
 with Path(path).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def write_csv(path,rows):
 if not rows: raise ValueError(f"No rows: {path}")
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def import_runtime(project_root):
 del project_root
 from .baseline_runtime import build_dataset,build_model,load_config,seed_everything
 from .evaluate_flame3_manual_smoke_v2 import extract_main_logits,extract_sample_keys
 return {"build_dataset":build_dataset,"build_model":build_model,"load_config":load_config,"seed_everything":seed_everything,"extract_main_logits":extract_main_logits,"extract_sample_keys":extract_sample_keys}


def checkpoint_path(root,arm,seed):
 if arm=="baseline": group="flame3_pidnet_s_fusion_newlabel_baseline_v2"; prefix="flame3_fusion_newlabel_baseline_v2_30e"
 else: group="flame3_pidnet_s_fusion_manual_smoke_v11"; prefix="flame3_fusion_manual_smoke_v11_30e"
 return root/"experiments"/group/f"{prefix}_seed{seed}"/"best_S.pth"


def config_path(root,arm):
 name="pidnet_s_fusion_newlabel_baseline_v2_30e.yaml" if arm=="baseline" else "pidnet_s_fusion_manual_smoke_v11_30e.yaml"
 return root/"configs"/"flame3"/name


def audit_checkpoint(path,arm,seed):
 actual=sha256_file(path); expected=EXPECTED_CHECKPOINT_SHA256[(arm,seed)]
 if actual!=expected: raise RuntimeError(f"Checkpoint hash mismatch: {path}")
 payload=torch.load(path,map_location="cpu",weights_only=False); cfg=payload.get("config",{})
 if int(cfg.get("SEED",-1))!=seed or cfg.get("MODEL")!="pidnet_s" or cfg.get("MODE")!="fusion": raise RuntimeError(f"Checkpoint identity mismatch: {path}")
 if bool(cfg.get("FLAME3_MANUAL_SMOKE_V2_ENABLED",False))!=(arm=="v11"): raise RuntimeError(f"Checkpoint arm mismatch: {path}")
 return {"arm":arm,"seed":seed,"path":str(path.resolve()),"sha256":actual,"epoch":int(payload.get("epoch",-1))}


def build_preflight(args):
 paths={"test_csv":args.test_csv.resolve(),"strata":args.strata_manifest.resolve(),"prereg":args.preregistration.resolve(),"frozen":args.frozen_result.resolve()}
 expected={"test_csv":EXPECTED_TEST_CSV_SHA256,"strata":EXPECTED_STRATA_SHA256,"prereg":EXPECTED_PREREG_SHA256,"frozen":EXPECTED_FROZEN_RESULT_SHA256}
 for name,path in paths.items():
  if not path.is_file() or sha256_file(path)!=expected[name]: raise RuntimeError(f"Frozen input mismatch: {name} {path}")
 test_rows=read_csv(paths["test_csv"]); strata=read_csv(paths["strata"]); prereg=json.loads(paths["prereg"].read_text(encoding="utf-8")); frozen=json.loads(paths["frozen"].read_text(encoding="utf-8"))
 if len(test_rows)!=107 or len(strata)!=107 or len(prereg["perturbations"])!=29: raise RuntimeError("Frozen test/prereg counts changed")
 if {r["sample_key"] for r in test_rows}!={r["sample_key"] for r in strata}: raise RuntimeError("Strata/test sample keys differ")
 frozen_map={(r["arm"],int(r["seed"])):r for r in frozen["runs"]}
 checkpoints=[audit_checkpoint(checkpoint_path(args.project_root.resolve(),arm,seed),arm,seed) for arm in ("baseline","v11") for seed in EXPECTED_SEEDS]
 if set(frozen_map)!={(a,s) for a in ("baseline","v11") for s in EXPECTED_SEEDS}: raise RuntimeError("Frozen result run identities changed")
 return {"protocol":"flame3_test107_posthoc_strata_perturbation_preflight","status":"passed","nature_disclosure":"The 107-image test has already undergone frozen evaluations. This run is post-hoc descriptive analysis and robustness checking of frozen checkpoints, not a new test-selection session.","paths":{k:{"path":str(v),"sha256":expected[k]} for k,v in paths.items()},"perturbation_count":29,"checkpoint_count":6,"batch_size":args.batch_size,"amp":bool(args.amp),"augment":False,"checkpoints":checkpoints,"training_performed":False,"threshold_tuning_performed":False,"predictions_saved":False,"method_identity_may_change":False}


def metrics(conf,nofire=None):
 out={}; ious=[]
 for i,name in enumerate(CLASS_NAMES):
  tp=int(conf[i,i]); fn=int(conf[i,:].sum()-tp); fp=int(conf[:,i].sum()-tp); den=tp+fp+fn
  iou=tp/den if den else 0.; precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
  out[f"{name}_iou"]=iou; out[f"{name}_precision"]=precision; out[f"{name}_recall"]=recall; ious.append(iou)
 out["three_class_miou"]=mean(ious); out["equal_fire_smoke_S"]=.5*(out["smoke_iou"]+out["fire_heat_iou"])
 if nofire is None: out["no_fire_joint_fp"]=0.
 else:
  valid=int(nofire.sum()); fp=int(nofire[:,1:].sum()); out["no_fire_joint_fp"]=fp/max(valid,1)
 return out


def stable_seed(sample_key,condition_id):
 return int.from_bytes(hashlib.sha256(f"{sample_key}|{condition_id}".encode()).digest()[:8],"little") & 0x7FFFFFFFFFFFFFFF


def gaussian_blur(channel,sigma):
 radius=max(1,int(math.ceil(3*sigma))); x=torch.arange(-radius,radius+1,device=channel.device,dtype=channel.dtype); k=torch.exp(-(x*x)/(2*sigma*sigma)); k=k/k.sum()
 y=F.conv2d(F.pad(channel,(radius,radius,0,0),mode="reflect"),k.view(1,1,1,-1)); return F.conv2d(F.pad(y,(0,0,radius,radius),mode="reflect"),k.view(1,1,-1,1))


def shift_channel(channel,pixels,direction):
 out=torch.zeros_like(channel); h,w=channel.shape[-2:]
 if direction=="left": out[:,:,:,:w-pixels]=channel[:,:,:,pixels:]
 elif direction=="right": out[:,:,:,pixels:]=channel[:,:,:,:w-pixels]
 elif direction=="up": out[:,:,:h-pixels,:]=channel[:,:,pixels:,:]
 elif direction=="down": out[:,:,pixels:,:]=channel[:,:,:h-pixels,:]
 else: raise ValueError(direction)
 return out


def perturb(images,names,condition):
 family=condition["family"]
 if family=="clean": return images
 out=images.clone()
 if family=="thermal_noise":
  sigma=float(condition["sigma"])
  for i,key in enumerate(names):
   gen=torch.Generator(device=out.device); gen.manual_seed(stable_seed(key,condition["id"])); noise=torch.randn(out[i,3].shape,generator=gen,device=out.device,dtype=out.dtype); out[i,3]=(out[i,3]+sigma*noise).clamp(0,1)
 elif family=="thermal_blur": out[:,3:4]=gaussian_blur(out[:,3:4],float(condition["sigma"]))
 elif family=="thermal_shift": out[:,3:4]=shift_channel(out[:,3:4],int(condition["pixels"]),str(condition["direction"]))
 elif family=="rgb_brightness": out[:,:3]=(out[:,:3]*float(condition["factor"])).clamp(0,1)
 elif family=="rgb_temperature":
  delta=float(condition["delta"]); out[:,0]=(out[:,0]*(1+delta)).clamp(0,1); out[:,2]=(out[:,2]*(1-delta)).clamp(0,1)
 elif condition["id"]=="thermal_zero": out[:,3:4]=0
 elif condition["id"]=="rgb_zero": out[:,:3]=0
 else: raise ValueError(condition)
 return out


def load_inference_model(runtime,cfg,payload,device):
 model=runtime["build_model"](cfg,augment=False); incompat=model.load_state_dict(payload["model_state_dict"],strict=False); missing=list(incompat.missing_keys); unexpected=list(incompat.unexpected_keys); allowed=("seghead_p.","seghead_d.")
 bad=[k for k in unexpected if not k.startswith(allowed)]
 if missing or bad or not unexpected: raise RuntimeError(f"augment=False load mismatch: missing={missing}, bad={bad}, unexpected_count={len(unexpected)}")
 return model.to(device).eval(),len(unexpected)


def matrix_from_row(row):
 return np.array([[int(row[f"c{i}{j}"]) for j in range(3)] for i in range(3)],dtype=np.int64)


def evaluate_run(runtime,args,record,strata_map,conditions,expected_conf):
 arm=record["arm"]; seed=int(record["seed"]); cfg=runtime["load_config"](config_path(args.project_root.resolve(),arm)); cfg["ROOTDATASET"]=str(args.bundle_root.resolve()); cfg["TESTSET"]=str(args.test_csv.resolve()); cfg["BATCHSIZE"]=args.batch_size; cfg["NUM_WORKERS"]=args.num_workers; cfg["DEVICE"]=args.device
 runtime["seed_everything"](seed); dataset=runtime["build_dataset"](cfg,split="test"); loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=args.device.startswith("cuda"),drop_last=False)
 payload=torch.load(record["path"],map_location="cpu",weights_only=False); device=torch.device(args.device); model,ignored=load_inference_model(runtime,cfg,payload,device)
 confs={c["id"]:np.zeros((3,3),dtype=np.int64) for c in conditions}; nofire={c["id"]:np.zeros((3,3),dtype=np.int64) for c in conditions}; per=[]; equivalence=None; seen=set()
 with torch.inference_mode():
  for batch in loader:
   images,labels=batch[0],batch[1]; names=runtime["extract_sample_keys"](batch[3]); flags=batch[4].detach().cpu().numpy().astype(bool).reshape(-1); images=images.to(device=device,dtype=torch.float); targets=labels.numpy().astype(np.uint8)
   if equivalence is None:
    compat=runtime["build_model"](cfg,augment=True); compat.load_state_dict(payload["model_state_dict"],strict=True); compat.to(device).eval(); a=runtime["extract_main_logits"](compat(images)); b=runtime["extract_main_logits"](model(images)); equivalence=float((a-b).abs().max().item())
    if equivalence>=1e-6 or not torch.equal(a.argmax(1),b.argmax(1)): raise RuntimeError(f"augment equivalence failed: {arm} {seed} {equivalence}")
    del compat,a,b
   for condition in conditions:
    x=perturb(images,names,condition)
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(args.amp and device.type=="cuda")):
     logits=runtime["extract_main_logits"](model(x)); logits=F.interpolate(logits,size=labels.shape[-2:],mode="bilinear",align_corners=bool(cfg.get("ALIGN_CORNERS",True))) if logits.shape[-2:]!=labels.shape[-2:] else logits
    preds=logits.argmax(1).cpu().numpy().astype(np.uint8)
    for i,key in enumerate(names):
     if condition["id"]=="clean":
      if key in seen: raise RuntimeError(f"Duplicate key {key}")
      seen.add(key)
     target=targets[i]; pred=preds[i]; valid=target!=255; enc=target[valid].astype(np.int64)*3+pred[valid].astype(np.int64); mat=np.bincount(enc,minlength=9).reshape(3,3); confs[condition["id"]]+=mat
     if not bool(flags[i]): nofire[condition["id"]]+=mat
     meta=strata_map[key]; row={"arm":arm,"seed":seed,"condition_id":condition["id"],"condition_family":condition["family"],"sample_key":key,"annotation_id":meta["annotation_id"],"sample_class":meta["sample_class"],"time_position_bin":meta["time_position_bin"],"smoke_coverage_bin":meta["smoke_coverage_bin"],"fire_heat_scale_bin":meta["fire_heat_scale_bin"],"brightness_bin":meta["brightness_bin"],"valid_pixels":int(valid.sum()),"ignore_pixels":int((~valid).sum())}
     for ii in range(3):
      for jj in range(3): row[f"c{ii}{jj}"]=int(mat[ii,jj])
     per.append(row)
 if len(seen)!=107: raise RuntimeError(f"Incomplete clean evaluation: {len(seen)}")
 if not np.array_equal(confs["clean"],expected_conf): raise RuntimeError(f"Clean confusion does not reproduce frozen test result: {arm} {seed}; delta={(confs['clean']-expected_conf).tolist()}")
 condition_rows=[]
 for condition in conditions:
  met=metrics(confs[condition["id"]],nofire[condition["id"]]); condition_rows.append({"arm":arm,"seed":seed,"condition_id":condition["id"],"condition_family":condition["family"],"condition_parameters":json.dumps({k:v for k,v in condition.items() if k not in ("id","family")},sort_keys=True),**met})
 clean=[r for r in per if r["condition_id"]=="clean"]; strata_rows=[]
 for field in ("time_position_bin","smoke_coverage_bin","fire_heat_scale_bin","brightness_bin","sample_class"):
  for value in sorted({r[field] for r in clean}):
   group=[r for r in clean if r[field]==value]; conf=sum((matrix_from_row(r) for r in group),np.zeros((3,3),dtype=np.int64)); nf=sum((matrix_from_row(r) for r in group if r["sample_class"]=="No Fire"),np.zeros((3,3),dtype=np.int64)); strata_rows.append({"arm":arm,"seed":seed,"stratum_type":field,"stratum_value":value,"sample_count":len(group),**metrics(conf,nf)})
 return {"condition_rows":condition_rows,"per_image_rows":per,"strata_rows":strata_rows,"equivalence":{"arm":arm,"seed":seed,"ignored_auxiliary_key_count":ignored,"fp32_max_abs_error":equivalence,"clean_confusion_exactly_matches_frozen_result":True}}


def aggregate(condition_rows):
 metric_names=list(METRIC_DIRECTIONS); agg=[]; paired=[]; degradation=[]
 by={(r["arm"],int(r["seed"]),r["condition_id"]):r for r in condition_rows}; condition_ids=sorted({r["condition_id"] for r in condition_rows})
 for arm in ("baseline","v11"):
  for cid in condition_ids:
   rs=[by[(arm,s,cid)] for s in EXPECTED_SEEDS]; row={"arm":arm,"condition_id":cid,"condition_family":rs[0]["condition_family"]}
   for m in metric_names:
    vals=[float(r[m]) for r in rs]; row[f"{m}_mean"]=mean(vals); row[f"{m}_sd"]=stdev(vals)
   agg.append(row)
 for cid in condition_ids:
  row={"condition_id":cid,"condition_family":by[("baseline",200,cid)]["condition_family"]}
  for m,direction in METRIC_DIRECTIONS.items():
   vals=[float(by[("v11",s,cid)][m])-float(by[("baseline",s,cid)][m]) for s in EXPECTED_SEEDS]; row[f"delta_{m}_mean"]=mean(vals); row[f"delta_{m}_sd"]=stdev(vals); row[f"improved_seed_count_{m}"]=sum(v*direction>0 for v in vals)
  paired.append(row)
 for arm in ("baseline","v11"):
  for cid in condition_ids:
   if cid=="clean": continue
   row={"arm":arm,"condition_id":cid,"condition_family":by[(arm,200,cid)]["condition_family"]}
   for m in metric_names:
    vals=[float(by[(arm,s,cid)][m])-float(by[(arm,s,"clean")][m]) for s in EXPECTED_SEEDS]; row[f"delta_from_clean_{m}_mean"]=mean(vals); row[f"delta_from_clean_{m}_sd"]=stdev(vals)
   degradation.append(row)
 return agg,paired,degradation


def build_curves(aggregate_rows,degradation_rows,conditions):
 agg={(r["arm"],r["condition_id"]):r for r in aggregate_rows}; deg={(r["arm"],r["condition_id"]):r for r in degradation_rows}; rows=[]
 for arm in ("baseline","v11"):
  clean=agg[(arm,"clean")]
  for family in ("thermal_noise","thermal_blur","thermal_shift","rgb_brightness","rgb_temperature","modality_failure"):
   family_conditions=[c for c in conditions if c["family"]==family]
   if family=="thermal_shift":
    for px in (1,2,4):
     cs=[c for c in family_conditions if int(c["pixels"])==px]; row={"arm":arm,"family":family,"level":px,"condition_ids":"|".join(c["id"] for c in cs),"replicate_count":len(cs)*3}
     for m in METRIC_DIRECTIONS:
      row[f"{m}_mean"]=mean(float(agg[(arm,c["id"])][f"{m}_mean"]) for c in cs); row[f"delta_from_clean_{m}_mean"]=mean(float(deg[(arm,c["id"])][f"delta_from_clean_{m}_mean"]) for c in cs)
     rows.append(row)
   else:
    for c in family_conditions:
     level=c.get("sigma",c.get("factor",c.get("delta",c["id"]))); row={"arm":arm,"family":family,"level":level,"condition_ids":c["id"],"replicate_count":3}
     for m in METRIC_DIRECTIONS: row[f"{m}_mean"]=agg[(arm,c["id"])][f"{m}_mean"]; row[f"delta_from_clean_{m}_mean"]=deg[(arm,c["id"])][f"delta_from_clean_{m}_mean"]
     rows.append(row)
 return rows


def report(aggregate_rows,paired_rows,degradation_rows):
 agg={(r["arm"],r["condition_id"]):r for r in aggregate_rows}; pair={r["condition_id"]:r for r in paired_rows}; worst=sorted([r for r in degradation_rows if r["arm"]=="v11"],key=lambda r:float(r["delta_from_clean_equal_fire_smoke_S_mean"]))[:8]
 lines=["# FLAME3 test107 post-hoc strata and perturbation analysis","","> The 107-image test set had already been evaluated under frozen protocols. This artifact is post-hoc descriptive analysis and robustness checking of frozen checkpoints, not a new test-selection session.","","## Clean equivalence","","All six clean confusion matrices exactly reproduce the frozen August 20 evaluation. Inference uses `augment=False`; only unused auxiliary-head checkpoint keys are ignored after strict prefix checks and FP32 main-logit equivalence tests.","","## Clean aggregate","","| Arm | Background IoU | Smoke IoU | Fire/Heat IoU | mIoU | S | No-Fire joint FP |","|---|---:|---:|---:|---:|---:|---:|"]
 for arm in ("baseline","v11"):
  r=agg[(arm,"clean")]; lines.append(f"| {arm} | {r['background_iou_mean']:.6f} | {r['smoke_iou_mean']:.6f} | {r['fire_heat_iou_mean']:.6f} | {r['three_class_miou_mean']:.6f} | {r['equal_fire_smoke_S_mean']:.6f} | {r['no_fire_joint_fp_mean']:.6f} |")
 p=pair["clean"]
 lines.extend([
  "",
  (
   "Paired v1.1-baseline clean deltas: "
   f"Smoke IoU {p['delta_smoke_iou_mean']:+.6f}, "
   f"Fire/Heat IoU {p['delta_fire_heat_iou_mean']:+.6f}, "
   f"mIoU {p['delta_three_class_miou_mean']:+.6f}, "
   f"S {p['delta_equal_fire_smoke_S_mean']:+.6f}."
  ),
  "",
  "## Largest v1.1 S degradation under perturbation",
  "",
  "| Condition | Family | Delta S from clean | Delta Fire IoU | Delta Smoke IoU |",
  "|---|---|---:|---:|---:|",
 ])
 for r in worst: lines.append(f"| {r['condition_id']} | {r['condition_family']} | {float(r['delta_from_clean_equal_fire_smoke_S_mean']):+.6f} | {float(r['delta_from_clean_fire_heat_iou_mean']):+.6f} | {float(r['delta_from_clean_smoke_iou_mean']):+.6f} |")
 lines.extend(["","Full per-image confusions, clean strata, condition aggregates, paired deltas, and degradation curves are provided as CSV files. No prediction images or tensors were saved.",""])
 return "\n".join(lines)


def main():
 args=parse_args(); runtime=import_runtime(args.project_root.resolve()); preflight=build_preflight(args); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
 if (output/"POSTHOC_COMPLETE.json").is_file(): print("ALREADY_COMPLETE"); return
 existing={p.name for p in output.iterdir()}; unexpected=existing.difference({"PREFLIGHT.json"})
 if unexpected and not args.preflight_only: raise FileExistsError(f"Refusing incomplete output: {sorted(unexpected)}")
 (output/"PREFLIGHT.json").write_text(json.dumps(preflight,indent=2),encoding="utf-8")
 if args.preflight_only: print(json.dumps(preflight,indent=2)); return
 strata_rows=read_csv(args.strata_manifest); strata_map={r["sample_key"]:r for r in strata_rows}; conditions=json.loads(args.preregistration.read_text(encoding="utf-8"))["perturbations"]; frozen=json.loads(args.frozen_result.read_text(encoding="utf-8")); expected={(r["arm"],int(r["seed"])):np.array(r["metrics"]["confusion_matrix_target_rows_prediction_columns"],dtype=np.int64) for r in frozen["runs"]}
 all_conditions=[]; all_per=[]; all_strata=[]; equivalence=[]
 for record in preflight["checkpoints"]:
  result=evaluate_run(runtime,args,record,strata_map,conditions,expected[(record["arm"],int(record["seed"]))]); all_conditions.extend(result["condition_rows"]); all_per.extend(result["per_image_rows"]); all_strata.extend(result["strata_rows"]); equivalence.append(result["equivalence"])
 aggregate_rows,paired_rows,degradation_rows=aggregate(all_conditions); curve_rows=build_curves(aggregate_rows,degradation_rows,conditions)
 write_csv(output/"CONDITION_METRICS_BY_RUN.csv",all_conditions); write_csv(output/"PER_IMAGE_CONFUSIONS.csv",all_per); write_csv(output/"CLEAN_STRATA_METRICS.csv",all_strata); write_csv(output/"CONDITION_THREE_SEED_AGGREGATE.csv",aggregate_rows); write_csv(output/"PAIRED_V11_MINUS_BASELINE.csv",paired_rows); write_csv(output/"DEGRADATION_FROM_CLEAN.csv",degradation_rows); write_csv(output/"PERTURBATION_CURVES.csv",curve_rows)
 summary={"protocol":"flame3_test107_posthoc_strata_and_perturbation","status":"complete","nature_disclosure":preflight["nature_disclosure"],"clean_equivalence":equivalence,"run_count":6,"perturbation_count":29,"per_image_confusion_rows":len(all_per),"strata_rows":len(all_strata),"training_performed":False,"threshold_tuning_performed":False,"predictions_saved":False,"method_identity_changed":False}
 (output/"POSTHOC_COMPLETE.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); (output/"POSTHOC_REPORT.md").write_text(report(aggregate_rows,paired_rows,degradation_rows),encoding="utf-8")
 audit={"status":"complete","clean_confusions_exactly_match_frozen_result":all(r["clean_confusion_exactly_matches_frozen_result"] for r in equivalence),"run_count":6,"perturbation_count":29,"per_image_confusion_rows":len(all_per),"result_sha256":sha256_file(output/"POSTHOC_COMPLETE.json"),"report_sha256":sha256_file(output/"POSTHOC_REPORT.md"),"training_performed":False,"threshold_tuning_performed":False,"predictions_saved":False}
 (output/"EVALUATION_AUDIT.json").write_text(json.dumps(audit,indent=2),encoding="utf-8"); print(json.dumps(audit,indent=2))

if __name__=="__main__": main()
