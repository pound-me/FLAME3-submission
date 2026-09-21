"""Frozen policy helpers for FLAME3 structure-screen stage 3."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ARMS = ("S3", "S4", "S6", "R1", "B1", "E1")
SEEDS = (200, 201, 202)
EPOCHS = (26, 27, 28, 29, 30)
CONDITIONS = (
    {"id": "clean", "family": "clean"},
    {"id": "thermal_noise_002", "family": "thermal_noise", "sigma": 0.02},
    {"id": "thermal_noise_005", "family": "thermal_noise", "sigma": 0.05},
    {"id": "thermal_noise_010", "family": "thermal_noise", "sigma": 0.10},
    {"id": "thermal_zero", "family": "zero"},
    {"id": "rgb_zero", "family": "zero"},
)
PROTOCOL_ID = "flame3_structure_stage3_readonly_20260919_v1"
SUPPLEMENT_SHA256 = "A2F962BC63875205B500FFC77F3D38D6BA13F72D702099B3817D28944C7EDC08"
PERTURB_SOURCE_SHA256 = "BE3761AFCB7EECDCEBDB7172CEA3BCDD664030DA5C57F6E94D95EC8E75911ACC"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows):
    if not rows:
        raise ValueError("No CSV rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Non-finite stage3 value")


def metric_view(validation):
    manual = validation["manual_smoke_v2"]["manual_A001_A047"]
    classes = manual["classes"]
    ious = [float(classes[name]["iou"]) for name in ("background", "smoke", "fire_heat")]
    return {
        "background_iou_A001_A047": ious[0],
        "smoke_iou_A001_A047": ious[1],
        "fire_heat_iou_A001_A047_record_only": ious[2],
        "three_class_miou_A001_A047_record_only": sum(ious) / 3.0,
        "selection_score_S_record_only": float(validation["selection_score_S"]),
        "fire_heat_iou_full134_record_only": float(validation["fire_heat_iou_full134"]),
        "no_fire_joint_false_positive_ratio": float(validation["no_fire_joint_false_positive_ratio"]),
        "smoke_precision_A001_A047": float(validation["smoke_precision_A001_A047"]),
        "smoke_recall_A001_A047": float(validation["smoke_recall_A001_A047"]),
    }


def average_views(views):
    if not views:
        raise ValueError("No metric views")
    return {key: sum(float(row[key]) for row in views) / len(views) for key in views[0]}


def clean_window_from_records(records):
    chosen = [row for row in records if int(row["epoch"]) in EPOCHS]
    if [int(row["epoch"]) for row in chosen] != list(EPOCHS):
        raise RuntimeError("Missing clean epoch26-30 window")
    return average_views([metric_view(row["validation"]) for row in chosen])


def summarize_decisions(clean_windows, robust_windows, baseline_clean, baseline_robust, efficiency):
    arms = {}
    for arm in ARMS:
        seed_rows = []
        for seed in SEEDS:
            candidate = clean_windows[arm][str(seed)]
            reference = baseline_clean[str(seed)]
            delta_smoke = candidate["smoke_iou_A001_A047"] - reference["smoke_iou_A001_A047"]
            delta_fp = (
                candidate["no_fire_joint_false_positive_ratio"]
                - reference["no_fire_joint_false_positive_ratio"]
            )
            guards = {
                "clean_smoke": candidate["smoke_iou_A001_A047"]
                >= reference["smoke_iou_A001_A047"] - 0.010,
                "no_fire_absolute": candidate["no_fire_joint_false_positive_ratio"] <= 0.005,
                "no_fire_increase": delta_fp <= 0.001,
            }
            candidate_robust = robust_windows[arm][str(seed)]
            base_epoch100 = baseline_robust[str(seed)]
            candidate_drop = (
                candidate_robust["clean"]["smoke_iou_A001_A047"]
                - candidate_robust["thermal_noise_005"]["smoke_iou_A001_A047"]
            )
            base_drop_epoch100 = (
                base_epoch100["clean"]["smoke_iou_A001_A047"]
                - base_epoch100["thermal_noise_005"]["smoke_iou_A001_A047"]
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "clean_smoke_delta": delta_smoke,
                    "no_fire_joint_fp_delta": delta_fp,
                    "guards": guards,
                    "precision_positive": delta_smoke >= 0.020,
                    "candidate_sigma005_drop": candidate_drop,
                    "baseline_epoch100_context_sigma005_drop": base_drop_epoch100,
                    "descriptive_robust_advantage_vs_baseline_epoch100_context": base_drop_epoch100 - candidate_drop,
                    "robust_rule_eligible": False,
                    "robust_rule_reason": "V1.1_EPOCH26_30_CHECKPOINTS_NOT_AVAILABLE",
                }
            )
        guards_3_of_3 = all(all(row["guards"].values()) for row in seed_rows)
        precision_3_of_3 = all(row["precision_positive"] for row in seed_rows)
        precision_pass = guards_3_of_3 and precision_3_of_3
        eff = efficiency[arm]
        efficiency_trigger = arm == "E1" and eff["gmacs_change_percent"] <= -5.0
        arms[arm] = {
            "seeds": seed_rows,
            "clean_guards_3_of_3": guards_3_of_3,
            "precision_rule_3_of_3": precision_3_of_3,
            "precision_axis_pass": precision_pass,
            "robust_axis_decision": "NOT_DETERMINABLE_BASELINE_WINDOW_MISSING",
            "efficiency": eff,
            "efficiency_proxy_trigger": efficiency_trigger,
            "efficiency_control_pass": bool(efficiency_trigger and guards_3_of_3),
            "formal_latency_status": "DEFERRED_BY_AUTHORIZATION",
            "overall_decision": (
                "PASS_PRECISION_AXIS"
                if precision_pass
                else (
                    "PASS_ENGINEERING_EFFICIENCY_CONTROL"
                    if efficiency_trigger and guards_3_of_3
                    else "NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES"
                )
            ),
            "S_used_for_decision": False,
            "Fire_used_for_positive_decision": False,
        }
    return arms
