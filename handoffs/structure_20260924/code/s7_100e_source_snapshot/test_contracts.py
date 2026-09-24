from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from audit_runtime import STAGEA_CANDIDATE_SHA, sha
from candidate_arms import ARMS, COUNTS, UAFMSpatial
from checkpointing import capture_rng, commit_epoch, load_committed, restore_rng
from common import EARLY_EPOCHS, EVAL_EPOCHS, SAVE_EPOCHS, SEEDS, TOTAL_EPOCHS
from reporting import csvout, decide, paired
from run_queue import SCHEDULE
from stage0 import ALLOWED_CONFIG_DIFFS, changed_keys


class Contracts(unittest.TestCase):
    def test_authorized_matrix_and_windows(self):
        self.assertEqual(ARMS, ('B0', 'S7'))
        self.assertEqual(SEEDS, (203, 204, 205))
        self.assertEqual(TOTAL_EPOCHS, 100)
        self.assertEqual(EARLY_EPOCHS, (26, 27, 28, 29, 30))
        self.assertEqual(EVAL_EPOCHS, (96, 97, 98, 99, 100))
        self.assertEqual(SCHEDULE, [('B0', 203), ('S7', 203), ('B0', 204), ('S7', 204), ('B0', 205), ('S7', 205)])

    def test_frozen_stagea_source_and_s7_shape(self):
        path = Path(__file__).with_name('stagea_candidate_arms_frozen.py')
        self.assertEqual(sha(path), STAGEA_CANDIDATE_SHA)
        module = UAFMSpatial().eval()
        p = torch.rand(2, 64, 64, 80)
        i = torch.rand(2, 64, 16, 20)
        output = module(p, i)
        self.assertEqual(tuple(output.shape), tuple(p.shape))
        self.assertEqual(COUNTS['S7'], (7712966, 7619810))

    def test_only_seed_and_epochs_are_config_differences(self):
        before = {'SEED': 200, 'EPOCHS': 30, 'LR': .001}
        after = {'SEED': 203, 'EPOCHS': 100, 'LR': .001}
        self.assertEqual(set(changed_keys(before, after)), ALLOWED_CONFIG_DIFFS)

    def test_decision_requires_all_three_seeds(self):
        baseline = {condition: {'smoke_iou_A001_A047': .70, 'no_fire_joint_false_positive_ratio': .0001}
                    for condition in ('clean', 'thermal_noise_002', 'thermal_noise_005', 'thermal_noise_010', 'thermal_zero', 'rgb_zero')}
        candidate = {condition: dict(values) for condition, values in baseline.items()}
        candidate['clean']['smoke_iou_A001_A047'] = .721
        candidate['thermal_noise_005']['smoke_iou_A001_A047'] = .721
        rows = [dict(seed=seed, **paired(candidate, baseline)) for seed in SEEDS]
        self.assertTrue(decide(rows)['precision_axis'])
        self.assertFalse(decide(rows)['robust_axis'])
        self.assertIsNone(decide(rows[:2])['precision_axis'])
        rows[2]['no_fire_relative'] = False
        rows[2]['precision_pass'] = False
        self.assertFalse(decide(rows)['precision_axis'])

    def test_delta_noise_identity(self):
        baseline = {condition: {'smoke_iou_A001_A047': .70, 'no_fire_joint_false_positive_ratio': .0001}
                    for condition in ('clean', 'thermal_noise_002', 'thermal_noise_005', 'thermal_noise_010', 'thermal_zero', 'rgb_zero')}
        candidate = {condition: dict(values) for condition, values in baseline.items()}
        candidate['clean']['smoke_iou_A001_A047'] = .71
        candidate['thermal_noise_005']['smoke_iou_A001_A047'] = .715
        row = paired(candidate, baseline)
        self.assertAlmostEqual(row['delta_noise005'], row['delta_clean'] + row['robust_gain'])

    def test_checkpoint_commit_keeps_current_and_previous(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            expected = dict(protocol='TEST', arm='B0', seed=203, source_manifest_sha256='SOURCE')
            for epoch in (1, 2, 3):
                record = {'epoch': epoch}
                payload = dict(**expected, epoch=epoch, epoch_record=record, tensor=torch.tensor([epoch]))
                commit_epoch(torch, folder, payload, record, SAVE_EPOCHS)
            pointer = __import__('audit_runtime').read(folder / 'COMMITTED_LAST.json')
            retained = sorted(path.name for path in (folder / 'recovery').glob('epoch*.pth'))
            self.assertEqual(pointer['epoch'], 3)
            self.assertEqual(retained, ['epoch002.pth', 'epoch003.pth'])
            self.assertTrue((folder / 'last.pth').exists())
            stale = folder / 'recovery' / 'epoch001.pth'
            stale.write_bytes((folder / 'recovery' / 'epoch002.pth').read_bytes())
            loaded, _, recovery = load_committed(torch, folder, expected, SAVE_EPOCHS)
            self.assertEqual(loaded['epoch'], 3)
            self.assertFalse(stale.exists())
            self.assertIsNotNone(recovery['stale_recovery_retention'])

    def test_access_audit_rotation_keeps_new_stream_writable(self):
        code = """
from pathlib import Path
import tempfile
import audit_runtime as ar
root = Path(tempfile.mkdtemp(prefix='flame3_audit_rotation_'))
ar.ROOT = root
ar.BUNDLE = root
worker = ar.AccessAudit([], 'first')
worker.rotate('second')
snapshot = worker.snapshot()
assert snapshot['phase'] == 'second'
assert len(list((root / 'audit' / 'access').glob('*.jsonl'))) == 2
"""
        result = subprocess.run(
            [sys.executable, '-B', '-c', code],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_csvout_accepts_registry_dictionary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'registry.csv'
            csvout(path, {'formal_started': [], 'prior_failed_attempts': [{'seed': 203}]})
            text = path.read_text(encoding='utf-8-sig')
            self.assertIn('formal_started', text)
            self.assertIn('prior_failed_attempts', text)

    def test_two_generator_rng_roundtrip(self):
        train = torch.Generator().manual_seed(203)
        validation = torch.Generator().manual_seed(100203)
        state = capture_rng(torch, train, validation)
        _ = torch.rand(3, generator=train)
        _ = torch.rand(3, generator=validation)
        restore_rng(torch, state, train, validation)
        self.assertTrue(torch.equal(train.get_state(), state['train_generator']))
        self.assertTrue(torch.equal(validation.get_state(), state['validation_generator']))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)