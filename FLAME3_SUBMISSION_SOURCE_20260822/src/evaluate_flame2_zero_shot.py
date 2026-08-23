from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

EXPECTED_SEEDS = (200, 201, 202)
EXPECTED_TEST_ROWS = 200
EXPECTED_TEST_SPLIT_SHA256 = "57e3a4b13be799809b5c013f028f76cecd64ae139caae5d15f9aabcf3fc8cb6a"
EXPECTED_CHECKPOINT_SHA256 = {
    ("baseline", 200): "1770dd6d94fe9bcee378c8a3674ad8dbd7feeb99b710befee6eedd060957f206",
    ("baseline", 201): "79bf3911aba25c5d484945735b9497ebbec3813feb174efb1f47e25543356a8e",
    ("baseline", 202): "b78e549e154316c404600d955b2f23af058397669fae61ddb95be33a6d2dd97f",
    ("v11", 200): "97e9d5902da9eb8beaa240760c41aa5e68086709ced6f8cb3f6481197a4cec6f",
    ("v11", 201): "32f7e5df88556d3efbee7227e31bfac19677691002a67c24a68e116474c799e7",
    ("v11", 202): "7e92ae0686b279c9f24f11d02bd8134e62454a10d489d079eb1d86262ca022cf",
}
CLASS_NAMES = ("background", "smoke", "fire_heat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_runtime(project_root: Path) -> dict[str, object]:
    del project_root
    from .baseline_runtime import build_model, load_config, seed_everything
    from .evaluate_flame3_manual_smoke_v2 import extract_main_logits, extract_sample_keys
    from third_party.RoboFireFuseNet.datasets.wildfire import WildFire
    return {
        "build_model": build_model,
        "load_config": load_config,
        "seed_everything": seed_everything,
        "WildFire": WildFire,
        "extract_main_logits": extract_main_logits,
        "extract_sample_keys": extract_sample_keys,
    }


def checkpoint_path(project_root: Path, arm: str, seed: int) -> Path:
    if arm == "baseline":
        group = "flame3_pidnet_s_fusion_newlabel_baseline_v2"
        prefix = "flame3_fusion_newlabel_baseline_v2_30e"
    elif arm == "v11":
        group = "flame3_pidnet_s_fusion_manual_smoke_v11"
        prefix = "flame3_fusion_manual_smoke_v11_30e"
    else:
        raise ValueError(arm)
    return project_root / "experiments" / group / f"{prefix}_seed{seed}" / "best_S.pth"


def config_path(project_root: Path, arm: str) -> Path:
    name = (
        "pidnet_s_fusion_newlabel_baseline_v2_30e.yaml"
        if arm == "baseline"
        else "pidnet_s_fusion_manual_smoke_v11_30e.yaml"
    )
    return project_root / "configs" / "flame3" / name


def audit_checkpoint(path: Path, arm: str, seed: int) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    expected = EXPECTED_CHECKPOINT_SHA256[(arm, seed)]
    if actual != expected:
        raise RuntimeError(f"Checkpoint hash mismatch: {path}: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored = payload.get("config", {})
    if int(stored.get("SEED", -1)) != seed:
        raise RuntimeError(f"Checkpoint seed mismatch: {path}")
    if stored.get("MODEL") != "pidnet_s" or stored.get("MODE") != "fusion":
        raise RuntimeError(f"Checkpoint architecture mismatch: {path}")
    manual = bool(stored.get("FLAME3_MANUAL_SMOKE_V2_ENABLED", False))
    if manual != (arm == "v11"):
        raise RuntimeError(f"Checkpoint arm mismatch: {path}")
    if arm == "v11" and float(stored.get("GRADIENT_CLIP_MAX_NORM", -1)) != 5.0:
        raise RuntimeError(f"Gradient clipping mismatch: {path}")
    return {
        "arm": arm,
        "seed": seed,
        "path": str(path.resolve()),
        "sha256": actual,
        "epoch": int(payload.get("epoch", -1)),
    }


def build_preflight(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    test_split = dataset_root / "lists" / "test_flm.txt"
    if not project_root.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("Project or FLAME2 dataset root missing")
    if sha256_file(test_split) != EXPECTED_TEST_SPLIT_SHA256:
        raise RuntimeError("FLAME2 test split hash changed")
    templates = [line.strip() for line in test_split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(templates) != EXPECTED_TEST_ROWS or len(set(templates)) != EXPECTED_TEST_ROWS:
        raise RuntimeError("FLAME2 test split must contain 200 unique rows")
    for template in templates:
        for modality in ("rgb", "ir", "gt"):
            path = dataset_root / template.replace("XXX", modality)
            if not path.is_file():
                raise FileNotFoundError(path)
    checkpoints = [
        audit_checkpoint(checkpoint_path(project_root, arm, seed), arm, seed)
        for arm in ("baseline", "v11")
        for seed in EXPECTED_SEEDS
    ]
    configs = {}
    for arm in ("baseline", "v11"):
        path = config_path(project_root, arm)
        if not path.is_file():
            raise FileNotFoundError(path)
        configs[arm] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return {
        "protocol": "flame3_frozen_checkpoint_flame2_zero_shot_preflight",
        "status": "passed",
        "source_dataset": "RoboFireFuseNet-preprocessed FLAME2 official test split",
        "source_test_rows": len(templates),
        "source_test_split_sha256": EXPECTED_TEST_SPLIT_SHA256,
        "class_mapping": {"0": "background", "1": "smoke", "2": "fire_heat"},
        "mapping_rationale": (
            "FLAME2 has an independent smoke class and visual audit shows class-2 regions "
            "under dense RGB smoke aligned with IR hotspots; class 2 is therefore not restricted "
            "to RGB-visible flame. No synthetic Ignore pixels are introduced."
        ),
        "coverage_limitation": (
            "FLAME2 class 2 cannot be proven to cover every residual-heat footprint in the broader "
            "FLAME3 Fire/Heat definition; results are reported as a coverage-mismatched lower-bound transfer estimate."
        ),
        "ir_preprocessing": "PIL convert('L'), divide by 255, matching the official WildFire loader",
        "spatial_preprocessing": "official FLAME2 256x256 deterministic resize/pad/crop; no augmentation",
        "model_augment_flag": False,
        "checkpoint_loading_rule": (
            "Load into augment=False model with strict=False, require no missing keys, "
            "and allow unexpected keys only under seghead_p./seghead_d.; verify FP32 main "
            "logits and argmax against a strict augment=True compatibility model."
        ),
        "checkpoints": checkpoints,
        "configs": configs,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "predictions_saved": False,
        "flame3_test_read": False,
    }


def confusion_metrics(confusion: np.ndarray) -> dict[str, object]:
    output: dict[str, object] = {}
    ious = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id, :].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        denom = tp + fp + fn
        iou = tp / denom if denom else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output[name] = {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
        ious.append(iou)
    output["three_class_mean_iou"] = mean(ious)
    output["equal_fire_smoke_S"] = 0.5 * (
        output["fire_heat"]["iou"] + output["smoke"]["iou"]
    )
    output["confusion_matrix_target_rows_prediction_columns"] = confusion.tolist()
    return output


def per_image_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    encoded = target.astype(np.int64).reshape(-1) * 3 + prediction.astype(np.int64).reshape(-1)
    confusion = np.bincount(encoded, minlength=9).reshape(3, 3)
    metrics = confusion_metrics(confusion)
    result: dict[str, object] = {
        "background_ratio": float((target == 0).mean()),
        "smoke_ratio": float((target == 1).mean()),
        "fire_heat_ratio": float((target == 2).mean()),
        "predicted_background_ratio": float((prediction == 0).mean()),
        "predicted_smoke_ratio": float((prediction == 1).mean()),
        "predicted_fire_heat_ratio": float((prediction == 2).mean()),
    }
    for name in CLASS_NAMES:
        result[f"{name}_iou"] = metrics[name]["iou"]
        result[f"{name}_precision"] = metrics[name]["precision"]
        result[f"{name}_recall"] = metrics[name]["recall"]
    result["three_class_mean_iou"] = metrics["three_class_mean_iou"]
    result["equal_fire_smoke_S"] = metrics["equal_fire_smoke_S"]
    return result


def evaluate(runtime: dict[str, object], args: argparse.Namespace, record: dict[str, object]) -> dict[str, object]:
    arm = str(record["arm"]); seed = int(record["seed"])
    cfg = runtime["load_config"](config_path(args.project_root.resolve(), arm))
    cfg["DEVICE"] = args.device
    runtime["seed_everything"](seed)
    dataset = runtime["WildFire"](
        root=str(args.dataset_root.resolve()),
        list_path="lists/test_flm.txt",
        num_classes=3,
        multi_scale=False,
        flip=False,
        brightness=False,
        ignore_label=255,
        base_size=256,
        crop_size=(256, 256),
        scale_factor=10,
        mean=[0, 0, 0, 0],
        std=[1, 1, 1, 1],
        mode="fusion",
        blend_images=False,
        comp_mask=False,
        single_source=False,
        seed=seed,
    )
    if len(dataset) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Unexpected FLAME2 test size: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    path = Path(str(record["path"]))
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Checkpoint changed after preflight: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = runtime["build_model"](cfg, augment=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_auxiliary_prefixes = ("seghead_p.", "seghead_d.")
    disallowed_unexpected = [
        key for key in unexpected
        if not key.startswith(allowed_auxiliary_prefixes)
    ]
    if missing or disallowed_unexpected or not unexpected:
        raise RuntimeError(
            "augment=False checkpoint loading mismatch: "
            f"missing={missing}, disallowed_unexpected={disallowed_unexpected}, "
            f"allowed_auxiliary_key_count={len(unexpected)}"
        )
    device = torch.device(args.device)
    model.to(device).eval()
    confusion = np.zeros((3, 3), dtype=np.int64)
    equivalence_checked = False
    equivalence_max_abs_error = None
    background_only_valid = 0
    background_only_false_positive = 0
    per_image_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with torch.inference_mode():
        for batch in loader:
            images, labels = batch[0], batch[1]
            names = runtime["extract_sample_keys"](batch[3])
            images = images.to(device=device, dtype=torch.float, non_blocking=True)
            if not equivalence_checked:
                compatibility_model = runtime["build_model"](cfg, augment=True)
                compatibility_model.load_state_dict(payload["model_state_dict"], strict=True)
                compatibility_model.to(device).eval()
                with torch.inference_mode():
                    reference_logits = runtime["extract_main_logits"](compatibility_model(images))
                    inference_logits = runtime["extract_main_logits"](model(images))
                equivalence_max_abs_error = float(
                    (reference_logits - inference_logits).abs().max().item()
                )
                if equivalence_max_abs_error >= 1e-6 or not torch.equal(
                    reference_logits.argmax(dim=1), inference_logits.argmax(dim=1)
                ):
                    raise RuntimeError(
                        "augment=True compatibility model and augment=False inference "
                        f"model are not equivalent: max_abs_error={equivalence_max_abs_error}"
                    )
                del compatibility_model, reference_logits, inference_logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                equivalence_checked = True
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(args.amp and device.type == "cuda"),
            ):
                logits = runtime["extract_main_logits"](model(images))
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=bool(cfg.get("ALIGN_CORNERS", True)),
                    )
            predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            targets = labels.numpy().astype(np.uint8)
            for index, name in enumerate(names):
                if name in seen:
                    raise RuntimeError(f"Duplicate FLAME2 sample: {name}")
                seen.add(name)
                target = targets[index]; prediction = predictions[index]
                if np.any(target == 255):
                    raise RuntimeError(f"Unexpected Ignore in official FLAME2 label: {name}")
                encoded = target.astype(np.int64).reshape(-1) * 3 + prediction.astype(np.int64).reshape(-1)
                confusion += np.bincount(encoded, minlength=9).reshape(3, 3)
                is_background_only = bool(np.all(target == 0))
                if is_background_only:
                    background_only_valid += int(target.size)
                    background_only_false_positive += int(np.count_nonzero(prediction != 0))
                row = {
                    "arm": arm,
                    "seed": seed,
                    "sample_name": name,
                    "background_only": is_background_only,
                    **per_image_metrics(target, prediction),
                }
                per_image_rows.append(row)
    if len(seen) != EXPECTED_TEST_ROWS:
        raise RuntimeError("Incomplete FLAME2 test evaluation")
    metrics = confusion_metrics(confusion)
    metrics["background_only_joint_false_positive_ratio"] = (
        background_only_false_positive / background_only_valid
        if background_only_valid else 0.0
    )
    metrics["background_only_valid_pixels"] = background_only_valid
    metrics["ignore_pixel_ratio"] = 0.0
    return {
        "arm": arm,
        "seed": seed,
        "checkpoint": record,
        "sample_count": len(seen),
        "augment_false_checkpoint_loading": {
            "missing_keys": missing,
            "ignored_auxiliary_checkpoint_key_count": len(unexpected),
            "ignored_auxiliary_prefixes": list(allowed_auxiliary_prefixes),
            "fp32_main_logits_equivalence_max_abs_error": equivalence_max_abs_error,
            "argmax_equivalent": True,
        },
        "metrics": metrics,
        "per_image": per_image_rows,
    }


def metric_value(run: dict[str, object], name: str) -> float:
    metrics = run["metrics"]
    if name.endswith("_iou") and name != "three_class_mean_iou":
        return float(metrics[name[:-4]]["iou"])
    if name.endswith("_precision"):
        return float(metrics[name[:-10]]["precision"])
    if name.endswith("_recall"):
        return float(metrics[name[:-7]]["recall"])
    return float(metrics[name])


def aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    names = (
        "background_iou", "smoke_iou", "smoke_precision", "smoke_recall",
        "fire_heat_iou", "fire_heat_precision", "fire_heat_recall",
        "three_class_mean_iou", "equal_fire_smoke_S",
        "background_only_joint_false_positive_ratio", "ignore_pixel_ratio",
    )
    by_arm = {
        arm: sorted([run for run in runs if run["arm"] == arm], key=lambda run: run["seed"])
        for arm in ("baseline", "v11")
    }
    summary: dict[str, object] = {"by_arm": {}, "paired_v11_minus_baseline": {}}
    for arm, arm_runs in by_arm.items():
        summary["by_arm"][arm] = {}
        for name in names:
            values = [metric_value(run, name) for run in arm_runs]
            summary["by_arm"][arm][name] = {
                "seed_200_201_202": values,
                "mean": mean(values),
                "sample_standard_deviation": stdev(values),
            }
    for name in names:
        values = [
            metric_value(by_arm["v11"][index], name) - metric_value(by_arm["baseline"][index], name)
            for index in range(3)
        ]
        summary["paired_v11_minus_baseline"][name] = {
            "seed_200_201_202": values,
            "mean": mean(values),
            "sample_standard_deviation": stdev(values),
            "positive_seed_count": sum(value > 0 for value in values),
        }
    return summary


def build_report(summary: dict[str, object]) -> str:
    by_arm = summary["by_arm"]; paired = summary["paired_v11_minus_baseline"]
    lines = [
        "# FLAME3 frozen checkpoints on FLAME2: zero-shot transfer",
        "",
        "> Read-only inference on the official 200-image FLAME2 test split. No training, fine-tuning, threshold selection, or prediction saving.",
        "",
        "## Frozen protocol",
        "",
        "- FLAME2 mapping: 0=Background, 1=Smoke, 2=Fire/Heat.",
        "- IR conversion: official loader `PIL.convert('L')`, then division by 255.",
        "- Spatial input: deterministic 256x256 official FLAME2 preprocessing.",
        "- Model inference: `augment=False`; six SHA256-pinned best_S checkpoints.",
        "- Coverage warning: FLAME2 Fire cannot be proven to cover every residual-heat footprint in the broader FLAME3 class; treat the numbers as a coverage-mismatched transfer estimate/lower bound.",
        "",
        "## Three-seed aggregate",
        "",
        "| Arm | Smoke IoU | Fire/Heat IoU | 3-class mIoU | Equal Fire-Smoke S | BG-only joint FP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("baseline", "v11"):
        a = by_arm[arm]
        def fmt(name: str) -> str:
            return f"{a[name]['mean']:.6f} +/- {a[name]['sample_standard_deviation']:.6f}"
        lines.append(f"| {arm} | {fmt('smoke_iou')} | {fmt('fire_heat_iou')} | {fmt('three_class_mean_iou')} | {fmt('equal_fire_smoke_S')} | {fmt('background_only_joint_false_positive_ratio')} |")
    lines.extend(["", "## Paired v1.1 minus baseline", "", "| Metric | Mean paired delta | Positive seeds |", "|---|---:|---:|"])
    for name in ("smoke_iou", "fire_heat_iou", "three_class_mean_iou", "equal_fire_smoke_S", "background_only_joint_false_positive_ratio"):
        item = paired[name]
        lines.append(f"| {name} | {item['mean']:+.6f} | {item['positive_seed_count']}/3 |")
    lines.extend(["", "This experiment is cross-dataset evidence only and must not be numerically merged with the FLAME3 validation or test tables.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runtime = import_runtime(args.project_root.resolve())
    preflight = build_preflight(args)
    output = args.output_dir.resolve()
    if output.exists() and not args.preflight_only:
        existing = {path.name for path in output.iterdir()}
        if "FLAME2_ZERO_SHOT_COMPLETE.json" in existing:
            print("ALREADY_COMPLETE")
            return
        unexpected = existing.difference({"PREFLIGHT.json"})
        if unexpected:
            raise FileExistsError(
                f"Refusing to overwrite incomplete output {output}: {sorted(unexpected)}"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "PREFLIGHT.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    runs = [evaluate(runtime, args, record) for record in preflight["checkpoints"]]
    all_per_image = [row for run in runs for row in run.pop("per_image")]
    write_csv(output / "PER_IMAGE_METRICS.csv", all_per_image)
    run_rows = []
    for run in runs:
        metrics = run["metrics"]
        run_rows.append({
            "arm": run["arm"], "seed": run["seed"], "checkpoint_sha256": run["checkpoint"]["sha256"],
            "background_iou": metrics["background"]["iou"], "smoke_iou": metrics["smoke"]["iou"],
            "smoke_precision": metrics["smoke"]["precision"], "smoke_recall": metrics["smoke"]["recall"],
            "fire_heat_iou": metrics["fire_heat"]["iou"], "fire_heat_precision": metrics["fire_heat"]["precision"],
            "fire_heat_recall": metrics["fire_heat"]["recall"], "three_class_mean_iou": metrics["three_class_mean_iou"],
            "equal_fire_smoke_S": metrics["equal_fire_smoke_S"],
            "background_only_joint_false_positive_ratio": metrics["background_only_joint_false_positive_ratio"],
            "ignore_pixel_ratio": metrics["ignore_pixel_ratio"],
        })
    write_csv(output / "RUN_METRICS.csv", run_rows)
    summary = aggregate(runs)
    result = {
        "protocol": "flame3_frozen_baseline_v11_on_flame2_zero_shot",
        "status": "complete",
        "preflight": preflight,
        "runs": runs,
        "aggregate": summary,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "predictions_saved": False,
        "flame3_test_read": False,
    }
    (output / "FLAME2_ZERO_SHOT_COMPLETE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "FLAME2_ZERO_SHOT_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    audit = {
        "status": "complete",
        "run_count": len(runs),
        "per_image_row_count": len(all_per_image),
        "source_test_rows": EXPECTED_TEST_ROWS,
        "result_sha256": sha256_file(output / "FLAME2_ZERO_SHOT_COMPLETE.json"),
        "report_sha256": sha256_file(output / "FLAME2_ZERO_SHOT_REPORT.md"),
        "training_performed": False,
        "predictions_saved": False,
        "flame3_test_read": False,
    }
    (output / "EVALUATION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
