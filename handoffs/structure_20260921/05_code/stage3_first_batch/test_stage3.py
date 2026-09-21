import unittest
from pathlib import Path

from stage3_protocol import ARMS, CONDITIONS, EPOCHS, SEEDS, summarize_decisions


def view(smoke=0.7, fp=0.0001):
    return {
        "background_iou_A001_A047": 0.8,
        "smoke_iou_A001_A047": smoke,
        "fire_heat_iou_A001_A047_record_only": 0.6,
        "three_class_miou_A001_A047_record_only": 0.7,
        "selection_score_S_record_only": 0.65,
        "fire_heat_iou_full134_record_only": 0.6,
        "no_fire_joint_false_positive_ratio": fp,
        "smoke_precision_A001_A047": 0.7,
        "smoke_recall_A001_A047": 0.7,
    }


class Stage3ContractTests(unittest.TestCase):
    def test_report_csv_is_allowlisted_but_not_hashed_as_input(self):
        source = (Path(__file__).resolve().parent / "run_stage3.py").read_text(encoding="utf-8")
        self.assertIn('report_outputs = {normalized(ROOT / "STAGE3_RESULTS.csv")}', source)
        self.assertIn("return allowed | report_outputs, current", source)

    def test_frozen_dimensions(self):
        self.assertEqual(len(ARMS) * len(SEEDS), 18)
        self.assertEqual(len(ARMS) * len(SEEDS) * len(EPOCHS) + len(SEEDS), 93)
        self.assertEqual(
            [row["id"] for row in CONDITIONS],
            ["clean", "thermal_noise_002", "thermal_noise_005", "thermal_noise_010", "thermal_zero", "rgb_zero"],
        )

    def test_s_and_fire_never_decide(self):
        clean = {arm: {str(seed): view(0.73) for seed in SEEDS} for arm in ARMS}
        baseline = {str(seed): view(0.70) for seed in SEEDS}
        robust = {
            arm: {str(seed): {condition["id"]: view(0.70) for condition in CONDITIONS} for seed in SEEDS}
            for arm in ARMS
        }
        base_robust = {
            str(seed): {condition["id"]: view(0.70) for condition in CONDITIONS}
            for seed in SEEDS
        }
        efficiency = {
            arm: {
                "deploy_parameters": 1, "gmacs_proxy": 1.0,
                "gmacs_change_percent": -5.1 if arm == "E1" else 0.0,
            }
            for arm in ARMS
        }
        result = summarize_decisions(clean, robust, baseline, base_robust, efficiency)
        self.assertTrue(result["S3"]["precision_axis_pass"])
        self.assertFalse(result["S3"]["S_used_for_decision"])
        self.assertFalse(result["S3"]["Fire_used_for_positive_decision"])
        self.assertEqual(
            result["S3"]["robust_axis_decision"],
            "NOT_DETERMINABLE_BASELINE_WINDOW_MISSING",
        )


if __name__ == "__main__":
    unittest.main()
