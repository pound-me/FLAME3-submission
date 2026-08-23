from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SEEDS = (200, 201, 202)
ARMS = {
    "A": {
        "config": "pidnet_s_fusion_manual_smoke_v11_30e.yaml",
        "prefix": "conv1_random_30e",
    },
    "B": {
        "config": "pidnet_s_fusion_manual_smoke_v11_conv1_rgb_irmean_30e.yaml",
        "prefix": "conv1_rgb_irmean_30e",
    },
}
WINDOW_METRICS = (
    "selection_score_S",
    "fire_heat_iou_full134",
    "fire_heat_precision_full134",
    "fire_heat_recall_full134",
    "smoke_iou_A001_A047",
    "smoke_precision_A001_A047",
    "smoke_recall_A001_A047",
    "no_fire_joint_false_positive_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the frozen FLAME3 conv1 ablation.")
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--annotation-package", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_config(root: Path, arm: str) -> dict[str, Any]:
    path = root / "configs" / ARMS[arm]["config"]
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def run_directory(root: Path, arm: str, seed: int) -> Path:
    config = load_config(root, arm)
    name = f"{ARMS[arm]['prefix']}_seed{seed}"
    return root / "experiments" / str(config["EXPERIMENT_GROUP"]) / name


def epoch_window(directory: Path) -> dict[str, float]:
    path = directory / "metrics.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_epoch = {int(record["epoch"]): record for record in records}
    missing = [epoch for epoch in range(26, 31) if epoch not in by_epoch]
    if missing:
        raise RuntimeError(f"{path} missing epochs {missing}")
    return {
        metric: statistics.fmean(
            float(by_epoch[epoch]["validation"][metric])
            for epoch in range(26, 31)
        )
        for metric in WINDOW_METRICS
    }


def evaluate_last_checkpoint(
    args: argparse.Namespace,
    root: Path,
    arm: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    directory = run_directory(root, arm, seed)
    checkpoint = directory / "last.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = output_dir / "per_run" / f"arm{arm}_seed{seed}_epoch30.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(checkpoint)
    if output.is_file():
        existing = load_json(output)
        if existing.get("checkpoint_sha256") == checkpoint_hash and existing.get("status") == "complete":
            return existing
        raise RuntimeError(f"Refusing to overwrite stale evaluation: {output}")
    command = [
        sys.executable,
        "-m",
        "src.evaluate_flame3_manual_smoke_v2",
        "--config",
        str(root / "configs" / ARMS[arm]["config"]),
        "--root-dataset",
        str(args.bundle_root.resolve()),
        "--val-csv",
        str(args.val_csv.resolve()),
        "--annotation-package",
        str(args.annotation_package.resolve()),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        args.device,
        "--batch-size",
        "8",
        "--num-workers",
        "0",
    ]
    if args.amp:
        command.append("--amp")
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"Evaluation failed: arm{arm} seed{seed}")
    result = load_json(output)
    if result.get("status") != "complete":
        raise RuntimeError(f"Evaluation incomplete: {output}")
    if result.get("checkpoint_epoch") != 30:
        raise RuntimeError(
            f"Primary endpoint requires epoch30 last.pth, got {result.get('checkpoint_epoch')}"
        )
    return result


def compact_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    manual = metrics["manual_A001_A047"]
    classes = manual["classes"]
    manual_miou = statistics.fmean(
        float(classes[name]["iou"])
        for name in ("background", "smoke", "fire_heat")
    )
    return {
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "manual_A001_A047_three_class_mIoU": manual_miou,
        "manual_A001_A047_classes": classes,
        "manual_A001_A047_smoke_to_fire_ratio": manual["smoke_to_fire_ratio"],
        "manual_A001_A047_fire_heat_to_smoke_ratio": manual["fire_heat_to_smoke_ratio"],
        "full_val134_fire_heat": metrics["full_134_fire_heat"],
        "selection_score_S": metrics["selection_score_S"],
        "no_fire": metrics["no_fire"],
    }


def numeric_delta(candidate: Any, reference: Any) -> Any:
    if isinstance(candidate, dict) and isinstance(reference, dict):
        return {
            key: numeric_delta(candidate[key], reference[key])
            for key in candidate.keys() & reference.keys()
            if isinstance(candidate[key], (dict, int, float))
            and isinstance(reference[key], (dict, int, float))
        }
    if isinstance(candidate, (int, float)) and isinstance(reference, (int, float)):
        return float(candidate) - float(reference)
    return None


def main() -> None:
    args = parse_args()
    root = args.submission_root.resolve()
    if "test" in args.val_csv.name.lower():
        raise RuntimeError("Test split is forbidden")
    prereg = load_json(root / "reproducibility" / "CONV1_INITIALIZATION_ABLATION_PREREGISTRATION.json")
    if prereg.get("status") != "frozen_before_training":
        raise RuntimeError("Preregistration is not frozen")
    output_dir = root / "results" / "conv1_initialization_ablation_30e"
    seeds: dict[str, Any] = {}
    primary_deltas: list[float] = []
    fire_deltas: list[float] = []
    fp_deltas: list[float] = []
    s_deltas: list[float] = []
    for seed in SEEDS:
        evaluated = {
            arm: compact_evaluation(
                evaluate_last_checkpoint(args, root, arm, seed, output_dir)
            )
            for arm in ("A", "B")
        }
        windows = {
            arm: epoch_window(run_directory(root, arm, seed))
            for arm in ("A", "B")
        }
        delta = numeric_delta(evaluated["B"], evaluated["A"])
        window_delta = {
            key: windows["B"][key] - windows["A"][key]
            for key in WINDOW_METRICS
        }
        primary = (
            evaluated["B"]["manual_A001_A047_three_class_mIoU"]
            - evaluated["A"]["manual_A001_A047_three_class_mIoU"]
        )
        fire_delta = (
            evaluated["B"]["full_val134_fire_heat"]["iou"]
            - evaluated["A"]["full_val134_fire_heat"]["iou"]
        )
        fp_delta = (
            evaluated["B"]["no_fire"]["joint_false_positive_ratio"]
            - evaluated["A"]["no_fire"]["joint_false_positive_ratio"]
        )
        s_delta = evaluated["B"]["selection_score_S"] - evaluated["A"]["selection_score_S"]
        primary_deltas.append(primary)
        fire_deltas.append(fire_delta)
        fp_deltas.append(fp_delta)
        s_deltas.append(s_delta)
        seeds[str(seed)] = {
            "A_random": evaluated["A"],
            "B_rgb_irmean": evaluated["B"],
            "B_minus_A": delta,
            "epoch26_30_window": {
                "A_random": windows["A"],
                "B_rgb_irmean": windows["B"],
                "B_minus_A": window_delta,
            },
        }
    means = {
        "manual_A001_A047_three_class_mIoU": statistics.fmean(primary_deltas),
        "full_val134_fire_heat_iou": statistics.fmean(fire_deltas),
        "no_fire_joint_false_positive_ratio": statistics.fmean(fp_deltas),
        "selection_score_S": statistics.fmean(s_deltas),
    }
    positive = sum(value > 0 for value in primary_deltas)
    guards_clear = (
        means["full_val134_fire_heat_iou"] >= float(prereg["guards"]["mean_full134_fire_heat_iou_delta_min"])
        and means["no_fire_joint_false_positive_ratio"] <= float(prereg["guards"]["mean_no_fire_joint_fp_delta_max"])
    )
    success = means["manual_A001_A047_three_class_mIoU"] > 0 and positive >= 2 and guards_clear
    decision = "adopt_rgb_copy_ir_mean_initialization" if success else "retain_random_four_channel_stem"
    result = {
        "protocol": "flame3_conv1_initialization_ablation_epoch30_last",
        "status": "complete",
        "preregistration_sha256": sha256_file(root / "reproducibility" / "CONV1_INITIALIZATION_ABLATION_PREREGISTRATION.json"),
        "primary_endpoint": "manual_A001_A047_three_class_mIoU",
        "checkpoint_rule": "epoch30_last.pth_for_both_arms",
        "seeds": seeds,
        "paired_deltas": {
            "primary": primary_deltas,
            "full_val134_fire_heat_iou": fire_deltas,
            "no_fire_joint_false_positive_ratio": fp_deltas,
            "selection_score_S": s_deltas,
        },
        "mean_paired_deltas": means,
        "sample_sd_paired_deltas": {
            "primary": statistics.stdev(primary_deltas),
            "full_val134_fire_heat_iou": statistics.stdev(fire_deltas),
            "no_fire_joint_false_positive_ratio": statistics.stdev(fp_deltas),
            "selection_score_S": statistics.stdev(s_deltas),
        },
        "positive_seed_count_primary": positive,
        "guards_clear": guards_clear,
        "decision": decision,
        "test_split_read": False,
        "training_protocol_modified_after_preregistration": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "CONV1_INITIALIZATION_ABLATION_COMPLETE.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FLAME3 four-channel conv1 initialization ablation",
        "",
        "Primary endpoint: epoch-30 `last.pth` on the frozen A001-A047 manual three-class development validation set. FLAME3 test107 was not read.",
        "",
        "| Seed | Delta manual 3-class mIoU | Delta Fire/Heat IoU (val134) | Delta S | Delta No-Fire joint FP |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, seed in enumerate(SEEDS):
        lines.append(
            f"| {seed} | {primary_deltas[index]:+.6f} | {fire_deltas[index]:+.6f} | "
            f"{s_deltas[index]:+.6f} | {fp_deltas[index]:+.6f} |"
        )
    lines.extend([
        f"| **Mean** | **{means['manual_A001_A047_three_class_mIoU']:+.6f}** | **{means['full_val134_fire_heat_iou']:+.6f}** | **{means['selection_score_S']:+.6f}** | **{means['no_fire_joint_false_positive_ratio']:+.6f}** |",
        "",
        f"Decision: `{decision}`; positive seeds: {positive}/3; guards clear: {guards_clear}.",
    ])
    (output_dir / "CONV1_INITIALIZATION_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "mean_paired_deltas": means, "positive_seed_count": positive, "guards_clear": guards_clear}, ensure_ascii=False))


if __name__ == "__main__":
    main()
