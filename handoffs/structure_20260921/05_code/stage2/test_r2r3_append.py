"""Contracts for the final R2/R3 tolerance and append queue."""
from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'stage1'))
sys.path.insert(0,str(HERE.parent/'stage1/implementation'))
from r2r3_protocol import ARMS,SEEDS,PROTOCOL_ID
from recheck_r2_r3_revised_gate import THRESHOLDS


class R2R3AppendContracts(unittest.TestCase):
    def test_final_thresholds_are_exact(self):
        self.assertEqual(THRESHOLDS['cpu_fp32'],{'head':1e-4,'stem':1e-6})
        self.assertEqual(THRESHOLDS['cuda_fp32'],{'head':1e-4,'stem':1e-6})
        self.assertEqual(THRESHOLDS['cuda_amp_fp16'],{'head':1e-2,'stem':1e-3})

    def test_cpu_evidence_passes_strictly(self):
        packaged=HERE/'revised_evidence/cpu/SUMMARY.json'
        path=(packaged if packaged.exists() else
              HERE.parent/'stage1/checks/r2_r3_revised_gate_cpu_attempt02_20260918/SUMMARY.json')
        summary=json.loads(path.read_text(encoding='utf-8'))
        self.assertTrue(summary['all_pass'])
        self.assertEqual({r['arm'] for r in summary['results']},set(ARMS))
        for result in summary['results']:
            check=result['precision_checks'][0]
            self.assertLess(check['maximum_head_abs'],1e-4)
            self.assertLess(check['maximum_stem_abs'],1e-6)
            signal=result['learning_signal']
            self.assertEqual(signal['initial_thermal_perturb_gate_max_abs'],0)
            self.assertGreater(signal['gate_final_bias_gradient_max_abs'],0)
            self.assertGreater(signal['post_one_synthetic_gate_step_thermal_perturb_max_abs'],0)

    def test_append_scope(self):
        self.assertEqual(ARMS,('R2','R3'))
        self.assertEqual(SEEDS,(200,201,202))
        self.assertIn('r2r3_append',PROTOCOL_ID)

    def test_r3_uses_thermal_loader_only(self):
        source=(HERE/'run_stage2_r2r3.py').read_text(encoding='utf-8')
        self.assertIn("if arm=='R3' else train_loader",source)
        self.assertNotIn("if arm=='R1' else train_loader",source)

    def test_append_waits_for_shared_gpu_lock(self):
        source=(HERE/'launch_r2r3_append.py').read_text(encoding='utf-8')
        self.assertIn('while LOCK.exists()',source)
        self.assertIn("with LOCK.open('x'",source)
        self.assertIn("'--shared-desktop'",source)
        self.assertIn("project_support_night_20260824/.deps",source)


if __name__=='__main__':
    unittest.main(verbosity=2)
