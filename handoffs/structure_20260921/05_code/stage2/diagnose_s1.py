"""Synthetic S1 arithmetic diagnosis only; never produces training eligibility."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'project_support_night_20260824/.deps'))
sys.path.insert(0,str(ROOT/'implementation'))
import torch
from engineering_checks import runtime,load_baseline,write_json
from structure_arms import apply_arm,deployment_copy
from stage2_protocol import verify_package


def main():
    torch.set_num_threads(4)
    verify_package(ROOT)
    out=ROOT/'diagnostics/S1_NUMERIC_DIAGNOSIS.json'
    if out.exists():
        raise FileExistsError(out)
    rt,_,_=runtime(ROOT/'source')
    baseline,_,identity=load_baseline(rt,ROOT/'source',ROOT.parent/'weights/PIDNet_S_ImageNet.pth.tar','cuda:0')
    candidate=apply_arm(baseline,'S1').cuda().eval()
    folded=deployment_copy(candidate,'S1').cuda().eval()
    rng=torch.Generator().manual_seed(20260917)
    images=[torch.rand(1,4,512,640,generator=rng),torch.zeros(1,4,512,640),torch.ones(1,4,512,640)]
    original_tf32=torch.backends.cudnn.allow_tf32
    records=[]
    try:
        for label,tf32,amp in (('original_fp32',original_tf32,False),('diagnostic_tf32_off_fp32',False,False),('original_amp',original_tf32,True)):
            torch.backends.cudnn.allow_tf32=tf32
            for index,image in enumerate(images):
                captured={}
                def capture_before(_module,args):
                    captured['input']=args[0].detach()
                def capture_after(_module,args,output):
                    captured['output']=output.detach()
                h1=candidate.spp.scale_process.register_forward_pre_hook(capture_before)
                h2=candidate.spp.scale_process.register_forward_hook(capture_after)
                try:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                        a=candidate(image.cuda())[1]
                        b=folded(image.cuda())
                        local_fold=folded.spp.scale_process(captured['input'])
                    row=dict(mode=label,probe=index,cudnn_allow_tf32=tf32,amp=amp,
                             main_max_abs=float((a.float()-b.float()).abs().max()),
                             scale_process_max_abs=float((captured['output'].float()-local_fold.float()).abs().max()))
                    if label=='original_fp32':
                        # Re-fold in float64 to distinguish formula from floating-point accumulation.
                        block=copy.deepcopy(candidate.spp.scale_process).cpu().double().eval()
                        block_fold=copy.deepcopy(block).to_deploy()
                        with torch.inference_mode():
                            x=captured['input'].cpu().double()
                            row['local_refolded_float64_max_abs']=float((block(x)-block_fold(x)).abs().max())
                    records.append(row)
                    print(row,flush=True)
                finally:
                    h1.remove()
                    h2.remove()
    finally:
        torch.backends.cudnn.allow_tf32=original_tf32
    write_json(out,dict(status='DIAGNOSIS_ONLY_ORIGINAL_GATE_FAILURE_UNCHANGED',identity=identity,
                       training_epochs=0,dataset_files_read=0,test107_read=False,
                       eligible_for_training=False,original_threshold=1e-4,
                       cudnn_tf32_restored=torch.backends.cudnn.allow_tf32==original_tf32,results=records))


if __name__=='__main__':
    main()
