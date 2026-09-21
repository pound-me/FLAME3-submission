"""Pure policy functions, frozen before replay outcomes are observed."""
from __future__ import annotations

RECOVERY_TOLERANCE = 1e-6
RECOVERY_METRICS = ("smoke_iou_A001_A047", "no_fire_joint_false_positive_ratio")


def recovery_check(actual, historical):
    differences = {k: abs(actual[k] - historical[k]) for k in RECOVERY_METRICS}
    return {"pass": all(v < RECOVERY_TOLERANCE for v in differences.values()),
            "absolute_differences": differences, "strict_tolerance": RECOVERY_TOLERANCE}


def paired_row(candidate, reference):
    clean, base = candidate["clean"], reference["clean"]
    smoke_delta = clean["smoke_iou_A001_A047"] - base["smoke_iou_A001_A047"]
    fp_delta = clean["no_fire_joint_false_positive_ratio"] - base["no_fire_joint_false_positive_ratio"]
    drop = clean["smoke_iou_A001_A047"] - candidate["thermal_noise_005"]["smoke_iou_A001_A047"]
    base_drop = base["smoke_iou_A001_A047"] - reference["thermal_noise_005"]["smoke_iou_A001_A047"]
    guards = dict(clean_smoke=smoke_delta >= -.010,
                  no_fire_absolute=clean["no_fire_joint_false_positive_ratio"] <= .005,
                  no_fire_increase=fp_delta <= .001)
    return dict(clean_smoke_delta=smoke_delta, no_fire_fp_delta=fp_delta, guards=guards,
                sigma005_drop=drop, baseline_sigma005_drop=base_drop,
                drop_reduction=base_drop-drop, precision_threshold_met=smoke_delta >= .020,
                robust_threshold_met=base_drop-drop >= .030)


def interaction(b0, r1, r2, r3):
    return dict(R1_minus_B0=r1-b0, R2_minus_B0=r2-b0,
                R3_minus_R1=r3-r1, interaction=r3-r1-r2+b0)
