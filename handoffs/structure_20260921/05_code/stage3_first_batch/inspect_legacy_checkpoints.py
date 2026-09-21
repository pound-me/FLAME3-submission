"""Map legacy v1.1 checkpoints to their exact metrics.jsonl epoch without inference."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent
BASE = BUNDLE / "project_support/experiments/flame3_pidnet_s_fusion_manual_smoke_v11"


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def signature(validation):
    return {
        "selection_score_S": float(validation["selection_score_S"]),
        "smoke_iou_A001_A047": float(validation["smoke_iou_A001_A047"]),
        "fire_heat_iou_full134": float(validation["fire_heat_iou_full134"]),
        "no_fire_joint_false_positive_ratio": float(validation["no_fire_joint_false_positive_ratio"]),
    }


output = {"training_performed": False, "inference_performed": False, "seeds": {}}
for seed in (200, 201, 202):
    folder = BASE / f"flame3_fusion_manual_smoke_v11_30e_seed{seed}"
    metrics = rows(folder / "metrics.jsonl")
    seed_output = {}
    for name in ("last.pth", "best.pth", "best_S.pth", "best_lowest_no_fire_joint_fp.pth"):
        path = folder / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        embedded = signature(payload["validation_metrics"])
        candidates = []
        for row in metrics:
            current = signature(row["validation"])
            maximum = max(abs(current[key] - embedded[key]) for key in embedded)
            candidates.append({"epoch": int(row["epoch"]), "maximum_abs_difference": maximum})
        candidates.sort(key=lambda item: (item["maximum_abs_difference"], item["epoch"]))
        seed_output[name] = {
            "checkpoint_header_epoch": payload.get("epoch"),
            "config_seed": payload.get("config", {}).get("SEED"),
            "embedded_signature": embedded,
            "closest_log_epochs": candidates[:3],
            "exact_log_epoch_matches": [
                item["epoch"] for item in candidates if item["maximum_abs_difference"] == 0.0
            ],
        }
    output["seeds"][str(seed)] = seed_output

(ROOT / "LEGACY_CHECKPOINT_EPOCH_MAP.json").write_text(
    json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
