from __future__ import annotations

"""Frozen metric accumulator for FLAME3 manual Smoke supervision v2."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image


BACKGROUND_ID = 0
SMOKE_ID = 1
FIRE_HEAT_ID = 2
IGNORE_ID = 255
NUM_CLASSES = 3


def load_frozen_manual_dev_targets(
    annotation_package: str | Path,
) -> dict[str, np.ndarray]:
    package = Path(annotation_package).resolve()
    freeze = json.loads(
        (
            package
            / "final_incremental_audit_20260815"
            / "ANNOTATION_FINAL_FREEZE.json"
        ).read_text(encoding="utf-8")
    )
    if freeze.get("status") != "fully_frozen_for_baseline_and_manual_smoke_v1_training":
        raise RuntimeError("Manual Smoke annotation package is not fully frozen")
    if freeze.get("test_images_or_labels_read") is not False:
        raise RuntimeError("Annotation freeze does not prove the test pool remained sealed")
    manifest = json.loads(
        (package / "flame3_threeclass_annotation_manifest_150.json").read_text(
            encoding="utf-8"
        )
    )
    targets: dict[str, np.ndarray] = {}
    seen_ids: list[str] = []
    for item in manifest["items"]:
        if item["split"] != "val":
            continue
        annotation_id = str(item["annotation_id"])
        number = int(annotation_id[1:])
        if not 1 <= number <= 47:
            raise RuntimeError(f"Unexpected manual validation ID: {annotation_id}")
        sample_key = str(item["sample_key"])
        if sample_key in targets:
            raise RuntimeError(f"Duplicate manual validation sample: {sample_key}")
        target = np.asarray(
            Image.open(package / "completed_masks" / f"{annotation_id}.png"),
            dtype=np.uint8,
        )
        if target.shape != (512, 640):
            raise RuntimeError(f"Manual validation shape mismatch: {annotation_id}")
        targets[sample_key] = target
        seen_ids.append(annotation_id)
    if len(targets) != 47 or sorted(seen_ids) != [
        f"A{index:03d}" for index in range(1, 48)
    ]:
        raise RuntimeError("Frozen A001-A047 manual targets are incomplete")
    return targets


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def _class_metrics(confusion: np.ndarray, class_index: int) -> dict[str, float | int]:
    true_positive = int(confusion[class_index, class_index])
    false_positive = int(confusion[:, class_index].sum() - true_positive)
    false_negative = int(confusion[class_index, :].sum() - true_positive)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "iou": _safe_ratio(
            true_positive, true_positive + false_positive + false_negative
        ),
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "recall": _safe_ratio(true_positive, true_positive + false_negative),
        "f1": _safe_ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
    }


@dataclass
class Flame3ManualSmokeV2Accumulator:
    expected_full_val_images: int = 134
    expected_manual_dev_images: int = 47
    expected_no_fire_images: int = 35
    full_fire_tp: int = 0
    full_fire_fp: int = 0
    full_fire_fn: int = 0
    full_fire_tn: int = 0
    full_val_images: int = 0
    no_fire_images: int = 0
    no_fire_valid_pixels: int = 0
    no_fire_predicted_background_pixels: int = 0
    no_fire_predicted_smoke_pixels: int = 0
    no_fire_predicted_fire_heat_pixels: int = 0
    manual_confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    )
    manual_seen_keys: set[str] = field(default_factory=set)

    def update(
        self,
        predictions: torch.Tensor | np.ndarray,
        machine_labels: torch.Tensor | np.ndarray,
        sample_keys: Sequence[str],
        fire_folder_flags: torch.Tensor | np.ndarray | Sequence[bool],
        manual_targets: Mapping[str, np.ndarray],
    ) -> None:
        prediction_array = (
            predictions.detach().cpu().numpy()
            if isinstance(predictions, torch.Tensor)
            else np.asarray(predictions)
        )
        machine_array = (
            machine_labels.detach().cpu().numpy()
            if isinstance(machine_labels, torch.Tensor)
            else np.asarray(machine_labels)
        )
        flag_array = (
            fire_folder_flags.detach().cpu().numpy()
            if isinstance(fire_folder_flags, torch.Tensor)
            else np.asarray(fire_folder_flags)
        ).astype(bool).reshape(-1)
        if prediction_array.shape != machine_array.shape or prediction_array.ndim != 3:
            raise ValueError(
                f"Prediction/label shape mismatch: {prediction_array.shape}/{machine_array.shape}"
            )
        batch_size = prediction_array.shape[0]
        if len(sample_keys) != batch_size or flag_array.size != batch_size:
            raise ValueError("Batch metadata does not match predictions")
        if not set(int(value) for value in np.unique(prediction_array)).issubset(
            {BACKGROUND_ID, SMOKE_ID, FIRE_HEAT_ID}
        ):
            raise ValueError("Predictions contain invalid class IDs")

        for index, sample_key in enumerate(sample_keys):
            prediction = prediction_array[index].astype(np.uint8, copy=False)
            machine = machine_array[index].astype(np.uint8, copy=False)
            valid = machine != IGNORE_ID
            target_fire = machine == FIRE_HEAT_ID
            predicted_fire = prediction == FIRE_HEAT_ID
            self.full_fire_tp += int((target_fire & predicted_fire & valid).sum())
            self.full_fire_fp += int((~target_fire & predicted_fire & valid).sum())
            self.full_fire_fn += int((target_fire & ~predicted_fire & valid).sum())
            self.full_fire_tn += int((~target_fire & ~predicted_fire & valid).sum())
            self.full_val_images += 1

            if not bool(flag_array[index]):
                self.no_fire_images += 1
                self.no_fire_valid_pixels += int(valid.sum())
                self.no_fire_predicted_background_pixels += int(
                    ((prediction == BACKGROUND_ID) & valid).sum()
                )
                self.no_fire_predicted_smoke_pixels += int(
                    ((prediction == SMOKE_ID) & valid).sum()
                )
                self.no_fire_predicted_fire_heat_pixels += int(
                    ((prediction == FIRE_HEAT_ID) & valid).sum()
                )

            if sample_key in manual_targets:
                if sample_key in self.manual_seen_keys:
                    raise RuntimeError(f"Duplicate manual dev image: {sample_key}")
                target = np.asarray(manual_targets[sample_key], dtype=np.uint8)
                if target.shape != prediction.shape:
                    raise ValueError(f"Manual target shape mismatch: {sample_key}")
                target_values = set(int(value) for value in np.unique(target))
                if not target_values.issubset(
                    {BACKGROUND_ID, SMOKE_ID, FIRE_HEAT_ID, IGNORE_ID}
                ):
                    raise ValueError(
                        f"Manual target has invalid IDs {target_values}: {sample_key}"
                    )
                manual_valid = target != IGNORE_ID
                encoded = (
                    target[manual_valid].astype(np.int64) * NUM_CLASSES
                    + prediction[manual_valid].astype(np.int64)
                )
                self.manual_confusion += np.bincount(
                    encoded, minlength=NUM_CLASSES * NUM_CLASSES
                ).reshape(NUM_CLASSES, NUM_CLASSES)
                self.manual_seen_keys.add(sample_key)

    def finalize(self, require_complete: bool = True) -> dict[str, object]:
        if require_complete:
            expected = (
                self.expected_full_val_images,
                self.expected_manual_dev_images,
                self.expected_no_fire_images,
            )
            actual = (
                self.full_val_images,
                len(self.manual_seen_keys),
                self.no_fire_images,
            )
            if actual != expected:
                raise RuntimeError(f"Incomplete frozen validation: {actual} != {expected}")

        fire_metrics = {
            "true_positive": self.full_fire_tp,
            "false_positive": self.full_fire_fp,
            "false_negative": self.full_fire_fn,
            "true_negative": self.full_fire_tn,
            "iou": _safe_ratio(
                self.full_fire_tp,
                self.full_fire_tp + self.full_fire_fp + self.full_fire_fn,
            ),
            "precision": _safe_ratio(
                self.full_fire_tp, self.full_fire_tp + self.full_fire_fp
            ),
            "recall": _safe_ratio(
                self.full_fire_tp, self.full_fire_tp + self.full_fire_fn
            ),
            "f1": _safe_ratio(
                2 * self.full_fire_tp,
                2 * self.full_fire_tp + self.full_fire_fp + self.full_fire_fn,
            ),
        }
        class_names = ("background", "smoke", "fire_heat")
        manual_metrics = {
            name: _class_metrics(self.manual_confusion, index)
            for index, name in enumerate(class_names)
        }
        smoke_iou = float(manual_metrics["smoke"]["iou"])
        fire_iou = float(fire_metrics["iou"])
        joint_fp_pixels = (
            self.no_fire_predicted_smoke_pixels
            + self.no_fire_predicted_fire_heat_pixels
        )
        smoke_to_fire = _safe_ratio(
            int(self.manual_confusion[SMOKE_ID, FIRE_HEAT_ID]),
            int(self.manual_confusion[SMOKE_ID, :].sum()),
        )
        fire_to_smoke = _safe_ratio(
            int(self.manual_confusion[FIRE_HEAT_ID, SMOKE_ID]),
            int(self.manual_confusion[FIRE_HEAT_ID, :].sum()),
        )
        return {
            "metric_protocol": "flame3_manual_smoke_supervision_v2_equal_fire_smoke_score",
            "selection_score_S": 0.5 * fire_iou + 0.5 * smoke_iou,
            "full_134_fire_heat": fire_metrics,
            "manual_A001_A047": {
                "image_count": len(self.manual_seen_keys),
                "confusion_matrix_target_rows_prediction_columns": self.manual_confusion.tolist(),
                "classes": manual_metrics,
                "smoke_to_fire_ratio": smoke_to_fire,
                "fire_heat_to_smoke_ratio": fire_to_smoke,
                "seen_sample_keys": sorted(self.manual_seen_keys),
            },
            "no_fire": {
                "image_count": self.no_fire_images,
                "valid_pixels": self.no_fire_valid_pixels,
                "predicted_background_pixels": self.no_fire_predicted_background_pixels,
                "predicted_smoke_pixels": self.no_fire_predicted_smoke_pixels,
                "predicted_fire_heat_pixels": self.no_fire_predicted_fire_heat_pixels,
                "predicted_smoke_ratio": _safe_ratio(
                    self.no_fire_predicted_smoke_pixels, self.no_fire_valid_pixels
                ),
                "predicted_fire_heat_ratio": _safe_ratio(
                    self.no_fire_predicted_fire_heat_pixels, self.no_fire_valid_pixels
                ),
                "joint_false_positive_pixels": joint_fp_pixels,
                "joint_false_positive_ratio": _safe_ratio(
                    joint_fp_pixels, self.no_fire_valid_pixels
                ),
            },
            "counts": {
                "full_val_images": self.full_val_images,
                "manual_dev_images": len(self.manual_seen_keys),
                "no_fire_images": self.no_fire_images,
            },
            "complete_frozen_validation": (
                self.full_val_images == self.expected_full_val_images
                and len(self.manual_seen_keys) == self.expected_manual_dev_images
                and self.no_fire_images == self.expected_no_fire_images
            ),
        }


def paired_guard_decision(
    mean_delta_s: float,
    positive_seed_count: int,
    mean_delta_fire_iou: float,
    mean_delta_smoke_iou: float,
    mean_delta_no_fire_joint_fp: float,
    catastrophe_threshold: float = 0.03,
) -> dict[str, object]:
    if catastrophe_threshold not in {0.02, 0.03}:
        raise ValueError("Catastrophe threshold must be frozen at 0.03 or 0.02")
    guard_reasons: list[str] = []
    if mean_delta_fire_iou < -catastrophe_threshold:
        guard_reasons.append("fire_heat_iou_catastrophe")
    if mean_delta_smoke_iou < -catastrophe_threshold:
        guard_reasons.append("smoke_iou_catastrophe")
    if mean_delta_no_fire_joint_fp > 0.001:
        guard_reasons.append("no_fire_joint_false_positive_increase")
    guards_clear = not guard_reasons
    if not guards_clear or mean_delta_s <= 0.0 or positive_seed_count <= 1:
        decision = "stop"
    elif mean_delta_s >= 0.01:
        decision = "pass"
    else:
        decision = "edge_continue_both_arms_to_50"
    return {
        "decision": decision,
        "guards_clear": guards_clear,
        "guard_reasons": guard_reasons,
        "catastrophe_threshold": catastrophe_threshold,
        "mean_delta_S": mean_delta_s,
        "positive_seed_count": positive_seed_count,
        "mean_delta_fire_heat_iou": mean_delta_fire_iou,
        "mean_delta_smoke_iou": mean_delta_smoke_iou,
        "mean_delta_no_fire_joint_fp": mean_delta_no_fire_joint_fp,
    }
