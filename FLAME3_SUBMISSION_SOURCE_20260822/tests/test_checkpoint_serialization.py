from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.benchmark_efficiency import safe_checkpoint_load
from src.evaluate_flame3_manual_smoke_v2 import (
    safely_load_training_checkpoint as safely_load_evaluation_checkpoint,
)
from src.train_baseline_v11 import save_checkpoint, safely_load_training_checkpoint


class _StateOnlyScaler:
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"scale": torch.tensor(128.0)}


class RestrictedCheckpointSerializationTest(unittest.TestCase):
    def test_new_checkpoint_is_weights_only_loadable(self) -> None:
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(200)
        np.random.seed(200)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "safe_checkpoint.pth"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                _StateOnlyScaler(),
                criterion,
                generator,
                {"MODEL": "pidnet_s", "MODE": "fusion", "SEED": 200},
                epoch=1,
                validation_metrics={"selection_score_S": 0.25},
                best_selection_metric=0.25,
                selection_metric_name="selection_score_S",
            )

            training_payload = safely_load_training_checkpoint(checkpoint_path)
            benchmark_payload, loading_mode = safe_checkpoint_load(checkpoint_path)
            with mock.patch.object(torch.serialization, "safe_globals", None):
                pytorch_21_payload = safely_load_evaluation_checkpoint(checkpoint_path)

        self.assertEqual(training_payload["epoch"], 1)
        self.assertEqual(benchmark_payload["epoch"], 1)
        self.assertEqual(pytorch_21_payload["epoch"], 1)
        self.assertIn("weights_only", loading_mode)
        numpy_state = training_payload["numpy_random_state"]
        self.assertEqual(
            set(numpy_state),
            {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"},
        )
        self.assertIsInstance(numpy_state["keys"], torch.Tensor)
        self.assertEqual(numpy_state["keys"].dtype, torch.int64)

        restored = (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
        np.random.set_state(restored)
        observed = np.random.randint(0, 2**31, size=8, dtype=np.int64)
        np.random.seed(200)
        expected = np.random.randint(0, 2**31, size=8, dtype=np.int64)
        np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
