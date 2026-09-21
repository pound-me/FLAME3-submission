import ast
import json
import unittest
from pathlib import Path

from replay_policy import recovery_check, paired_row, interaction
import stage2_protocol as protocol


class ReplayContracts(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(protocol.ARMS, ("B0",))
        self.assertEqual(protocol.SEEDS, (200, 201, 202))

    def test_recovery_strict_boundary(self):
        reference = dict(smoke_iou_A001_A047=0., no_fire_joint_false_positive_ratio=0.)
        self.assertTrue(recovery_check(reference, reference)["pass"])
        self.assertFalse(recovery_check({**reference,"smoke_iou_A001_A047":1e-6}, reference)["pass"])

    def test_interaction(self):
        self.assertAlmostEqual(interaction(.5,.6,.55,.7)["interaction"], .05)

    def test_robust_gain_never_overrides_clean_guard(self):
        ref = dict(clean=dict(smoke_iou_A001_A047=.70,no_fire_joint_false_positive_ratio=.001),
                   thermal_noise_005=dict(smoke_iou_A001_A047=.40))
        cand = dict(clean=dict(smoke_iou_A001_A047=.65,no_fire_joint_false_positive_ratio=.001),
                    thermal_noise_005=dict(smoke_iou_A001_A047=.60))
        value = paired_row(cand,ref)
        self.assertTrue(value["robust_threshold_met"])
        self.assertFalse(value["guards"]["clean_smoke"])

    def test_no_test_access(self):
        with self.assertRaises(RuntimeError):
            protocol.validate_image_access('test107/data.png', set())

    def test_training_loop_not_reinvented(self):
        worker = (Path(__file__).parent/'run_baseline.py').read_text(encoding='utf-8')
        ast.parse(worker)
        self.assertIn("model=apply_arm(baseline,'R1',seed=seed)",worker)
        self.assertIn("if arm=='R1' else train_loader",worker)
        self.assertIn('range(start_epoch,30)',worker)
        self.assertIn('if epoch+1>=26:',worker)

    def test_actual_frozen_summary_schema(self):
        root=Path(__file__).parent
        first=json.loads((root/'HISTORICAL_STAGE3_SUMMARY.json').read_text(encoding='utf-8'))
        append=json.loads((root/'HISTORICAL_APPEND_SUMMARY.json').read_text(encoding='utf-8'))
        candidates=dict(first['robust_windows_epoch26_30'])
        candidates.update(append['append_robust_epoch26_30'])
        self.assertEqual(len(candidates),9)
        for values in candidates.values():
            for seed in (200,201,202):
                self.assertEqual(len(values[str(seed)]),6)
                same=paired_row(values[str(seed)],values[str(seed)])
                self.assertEqual(same['clean_smoke_delta'],0)


if __name__ == '__main__':
    unittest.main()
