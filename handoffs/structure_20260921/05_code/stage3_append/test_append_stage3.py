from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRST_STAGE3 = HERE.parent / "flame3_structure_stage3_20260919_v1"
sys.path.insert(0, str(FIRST_STAGE3))

spec = importlib.util.spec_from_file_location("append_stage3", HERE / "run_append_stage3.py")
module = importlib.util.module_from_spec(spec)


class AppendStage3Contracts(unittest.TestCase):
    def test_frozen_scope(self):
        source = (HERE / "run_append_stage3.py").read_text(encoding="utf-8")
        self.assertIn('ARMS = ("S1", "R2", "R3")', source)
        self.assertIn("Expected 45 append checkpoint units", source)
        self.assertIn('condition["id"] != "clean"', source)

    def test_2x2_formula_and_limit(self):
        source = (HERE / "run_append_stage3.py").read_text(encoding="utf-8")
        self.assertIn('"interaction": "Y_R3 - Y_R1 - Y_R2 + Y_B0"', source)
        self.assertIn("y_r3 - y_r1 - y_r2 + b0", source)
        self.assertIn('"formal_attribution_eligible": False', source)
        self.assertIn("V1.1_EPOCH26_30_CHECKPOINTS_NOT_AVAILABLE", source)

    def test_read_only_boundaries(self):
        source = (HERE / "run_append_stage3.py").read_text(encoding="utf-8")
        self.assertIn('"test107_read": False', source)
        self.assertIn('"training_performed": False', source)
        self.assertIn('"predictions_saved": False', source)
        self.assertIn('normalized(ROOT / "ATTRIBUTION_2X2.csv")', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
