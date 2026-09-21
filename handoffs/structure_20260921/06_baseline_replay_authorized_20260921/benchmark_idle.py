"""One B0/E1 latency comparison, only from the Windows idle scheduled task."""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'project_support_night_20260824/.deps'))
sys.path.insert(0,str(ROOT/'implementation'))
from stage2_protocol import read,write,sha,verify_package,install_guard

LOCK=ROOT.parent/'flame3_structure_screen_20260917_gpu.lock'


def gpu_snapshot():
    def query(args):
        p=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,check=True)
        return p.stdout.strip()
    return dict(gpu=query(['--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader']),
                compute_pids=query(['--query-compute-apps=pid','--format=csv,noheader,nounits']))


def external(snapshot):
    return [int(p) for p in snapshot['compute_pids'].splitlines() if p.strip().isdigit() and int(p)!=os.getpid()]


def main():
    final=ROOT/'LATENCY_REPORT.json'
    if final.exists() or (ROOT/'LATENCY_FAILURE.json').exists():
        return
    if '--windows-idle-task' not in sys.argv:
        raise RuntimeError('Only the idle scheduled task may start timing')
    verify_package(ROOT)
    q=ROOT/'QUEUE_STATUS.json'
    if not q.exists() or read(q)['status']!='COMPLETE_B0_TRAINING_AND_EVALUATION' or LOCK.exists():
        write(ROOT/'LATENCY_STATUS.json',dict(status='WAITING_BASELINE_QUEUE',test107_read=False))
        return
    for _ in range(3):
        snap=gpu_snapshot()
        if external(snap) or float(snap['gpu'].split(',')[0].strip().replace('%',''))>5:
            write(ROOT/'LATENCY_STATUS.json',dict(status='WAITING_GPU_IDLE',snapshot=snap))
            return
        time.sleep(5)
    try:
        with LOCK.open('x',encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        return
    try:
        install_guard(set())
        import torch
        import numpy as np
        from thop import profile
        from engineering_checks import runtime
        from structure_arms import apply_arm,deployment_copy
        rt,_,_=runtime(ROOT/'source')
        rt.seed_everything(200)
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.matmul.allow_tf32=False
        generator=torch.Generator().manual_seed(20260921)
        image=torch.rand(1,4,512,640,generator=generator).cuda()
        env=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                 gpu=torch.cuda.get_device_name(0),cudnn_benchmark=False,cudnn_deterministic=True,
                 cudnn_allow_tf32=True,matmul_allow_tf32=False)
        observations=[]
        results={}
        for arm,folder in [('B0',ROOT/'runs/B0/seed200'),
                           ('E1',ROOT.parent/'stage2_structure_20260918_v1/runs/E1/seed200')]:
            checkpoint=folder/'epoch30.pth'
            before=sha(checkpoint)
            if before!=read(folder/'RESULT.json')['checkpoint_sha256']['epoch30.pth']:
                raise RuntimeError('Timing checkpoint hash mismatch')
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            base=rt.build_model(payload['config'],augment=True)
            model=base if arm=='B0' else apply_arm(base,'E1',seed=200)
            model.load_state_dict(payload['model_state_dict'],strict=True)
            deployed=deployment_copy(model,arm).cuda().eval()
            del payload,base,model
            torch.cuda.empty_cache()
            with torch.inference_mode():
                macs,_=profile(deployed,inputs=(image,),verbose=False)
                def forward():
                    with torch.autocast('cuda',dtype=torch.float16):
                        output=deployed(image)
                    if output.shape!=(1,3,64,80):
                        raise RuntimeError('Unexpected deployment output shape')
                for _ in range(100):
                    forward()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings=[]
                trial_means=[]
                for trial in range(10):
                    snapshot=gpu_snapshot()
                    observations.append(dict(arm=arm,trial=trial,snapshot=snapshot))
                    if external(snapshot):
                        raise RuntimeError('Shared GPU detected during timing; do not retry for a faster result')
                    times=[]
                    for _ in range(200):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record(); forward(); end.record(); end.synchronize()
                        times.append(float(start.elapsed_time(end)))
                    timings.extend(times)
                    trial_means.append(statistics.fmean(times))
            result=dict(checkpoint=str(checkpoint),checkpoint_sha256=before,seed=200,epoch=30,
                        deploy_parameters=sum(p.numel() for p in deployed.parameters()),gmacs=macs/1e9,
                        mean_ms=statistics.fmean(timings),p95_ms=float(np.percentile(timings,95)),
                        median_ms=float(np.median(timings)),std_ms=float(np.std(timings)),
                        samples_ms=timings,per_trial_mean_ms=trial_means,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
            if sha(checkpoint)!=before:
                raise RuntimeError('Read-only timing checkpoint changed')
            results[arm]=result
            del deployed
            torch.cuda.empty_cache()
        final_snapshot=gpu_snapshot()
        if external(final_snapshot):
            raise RuntimeError('External compute at timing end')
        write(final,dict(status='COMPLETE',results=results,environment=env,
                         windows_idle_minutes_required=10,protocol=dict(batch=1,input=[1,4,512,640],augment=False,
                         amp=True,warmup=100,trials=10,iterations_per_trial=200),observations=observations,
                         mean_latency_change_percent=100*(results['E1']['mean_ms']/results['B0']['mean_ms']-1),
                         p95_latency_change_percent=100*(results['E1']['p95_ms']/results['B0']['p95_ms']-1),
                         formal_environment_idle_at_launch=True,test107_read=False,training_performed=False,
                         source_checkpoints_unchanged=True,predictions_saved=False))
        write(ROOT/'LATENCY_STATUS.json',dict(status='COMPLETE',finished_unix=time.time()))
    except BaseException as exc:
        write(ROOT/'LATENCY_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),
              automatic_retry=False,formal_result_eligible=False))
        raise
    finally:
        if LOCK.exists() and LOCK.read_text(encoding='utf-8')==str(os.getpid()):
            LOCK.unlink()


if __name__=='__main__':
    main()
