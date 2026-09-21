"""Focused tests for the formal-run identity, guard and checkpoint contracts."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'stage1/implementation'))

import numpy as np
import torch
from stage2_protocol import (ARMS,SEEDS,PROTOCOL_ID,effective_config,read,normalized,validate_image_access,
                             input_paths,descriptive_summary,assert_finite)
from run_stage2 import save_checkpoint,verify_resume_header,restore_rng
from engineering_checks import runtime
from structure_arms import apply_arm,assert_unchanged_state

PROJECT=Path('G:/py2')


class Stage2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.rt,_,_=runtime(PROJECT)
        cls.frozen=cls.rt.load_config(PROJECT/'configs/flame3/pidnet_s_fusion_manual_smoke_v11_30e.yaml')

    def test_config_only_declared_overrides(self):
        before=copy.deepcopy(self.frozen)
        for seed in SEEDS:
            config,diff=effective_config(self.frozen,Path('D:/approved_bundle'),seed)
            self.assertEqual(config['SEED'],seed)
            self.assertEqual(config['EPOCHS'],30)
            self.assertEqual(config['LR_TOTAL_EPOCHS'],100)
            self.assertEqual(diff['undeclared_differences'],[])
            self.assertFalse(diff['selection_applied'])
        self.assertEqual(before,self.frozen)
        with self.assertRaises(ValueError):
            effective_config(self.frozen,Path('D:/approved_bundle'),203)

    def test_only_six_arms_eighteen_runs(self):
        self.assertEqual(len([(a,s) for a in ARMS for s in SEEDS]),18)
        self.assertTrue(set(('S1','R2','R3','S2','S5','S7','S8')).isdisjoint(ARMS))

    def test_guard_denies_nonallowlisted_data(self):
        allowed={normalized('C:/Tmp/approved/train.png')}
        validate_image_access('C:/Tmp/approved/train.png',allowed)
        for path in ('C:/Tmp/other.png','C:/Tmp/test.csv','C:/Tmp/test107/stats.json','C:/Tmp/test_blind/manifest.json'):
            with self.assertRaises(RuntimeError):
                validate_image_access(path,allowed)

    def test_input_manifest_rejects_test_and_prediction(self):
        for field in ('corrected_rgb_path','raw_thermal_path','temperature_mask_path'):
            row=dict(corrected_rgb_path='train/rgb.png',raw_thermal_path='train/ir.jpg',temperature_mask_path='train/mask.png')
            row[field]='test107/forbidden.png'
            with self.assertRaises(RuntimeError):
                input_paths(Path('C:/Tmp/train.csv'),[row],Path('C:/Tmp'))

    def test_fixed_window_not_best_S(self):
        names=('smoke_iou_A001_A047','background_iou_A001_A047','three_class_miou_A001_A047',
               'no_fire_joint_false_positive_ratio','selection_score_S','fire_heat_iou_full134')
        rows=[dict(epoch=i,validation={name:i/100 for name in names}) for i in range(1,31)]
        rows[0]['validation']['selection_score_S']=.99
        result=descriptive_summary(rows)
        self.assertAlmostEqual(result['window_means']['smoke_iou_A001_A047'],.28)
        self.assertEqual(result['best_S_record_only']['epoch'],1)
        self.assertIsNone(result['passed'])
        self.assertFalse(result['S_used_for_decision'])

    def test_nonfinite_metrics_rejected(self):
        for bad in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(RuntimeError):
                assert_finite({'loss':[1.,bad]})

    def test_checkpoint_safe_load_momentum_rng(self):
        self.rt.seed_everything(200)
        model=torch.nn.Linear(4,2)
        opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        generator=torch.Generator().manual_seed(200)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        model(torch.ones(3,4)).sum().backward()
        opt.step()
        config={'test_only':True}
        with tempfile.TemporaryDirectory(dir='C:/Tmp') as tmp:
            path=Path(tmp)/'last.pth'
            save_checkpoint(path,model,opt,scaler,generator,config,'S3',200,1,'ABC',{'epoch':1})
            saved=torch.load(path,map_location='cpu',weights_only=True)
            verify_resume_header(saved,'S3',200,config,'ABC')
            self.assertTrue(saved['optimizer_state_dict']['state'])
            expected=torch.rand(5)
            expected_np=np.random.rand(5)
            restored=torch.nn.Linear(4,2)
            restored.load_state_dict(saved['model_state_dict'])
            restore_rng(saved['rng'],generator)
            self.assertTrue(torch.equal(expected,torch.rand(5)))
            self.assertTrue(np.array_equal(expected_np,np.random.rand(5)))
            for a,b in zip(model.parameters(),restored.parameters()):
                self.assertTrue(torch.equal(a,b))
            saved['engineering_only']=True
            with self.assertRaises(RuntimeError):
                verify_resume_header(saved,'S3',200,config,'ABC')

    def test_each_seed_initialization_and_rng_isolation(self):
        stem_values=[]
        for seed in SEEDS:
            self.rt.seed_everything(seed)
            baseline=self.rt.build_model(self.frozen,augment=True)
            stem_values.append(baseline.conv1[0].weight.detach().clone())
            for arm in ARMS:
                state=torch.get_rng_state().clone()
                candidate=apply_arm(baseline,arm,seed)
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
                assert_unchanged_state(baseline,candidate,arm)
                self.assertTrue(torch.equal(candidate.conv1[0].weight,baseline.conv1[0].weight))
        self.assertFalse(torch.equal(stem_values[0],stem_values[1]))
        self.assertFalse(torch.equal(stem_values[1],stem_values[2]))


if __name__=='__main__':
    unittest.main(verbosity=2)
