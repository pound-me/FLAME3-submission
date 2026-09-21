"""Read-only Stage 3 evaluation for S1/R2/R3 plus R1/R2/R3 2x2 attribution."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent
APPEND_STAGE2 = BUNDLE / "stage2_s1r2r3_append_20260919_v1"
FIRST_STAGE2 = BUNDLE / "stage2_structure_20260918_v1"
FIRST_STAGE3 = BUNDLE / "stage3_structure_20260919_v1"
DEPS = BUNDLE / "project_support_night_20260824/.deps"
sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(APPEND_STAGE2 / "implementation"))
sys.path.insert(0, str(APPEND_STAGE2))
sys.path.insert(0, str(FIRST_STAGE3))

import torch
from torch.utils.data import DataLoader

from engineering_checks import make_criterion, runtime
from r2r3_protocol import (
    ANNOTATION_REL, VAL_REL, input_paths, install_guard, normalized, sha as stage2_sha,
)
from stage3_protocol import (
    CONDITIONS, EPOCHS, PERTURB_SOURCE_SHA256, SEEDS, assert_finite,
    average_views, clean_window_from_records, metric_view, read_json, read_jsonl,
    sha256, write_csv, write_json,
)
from structure_arms import apply_arm

ARMS = ("S1", "R2", "R3")
PROTOCOL_ID = "flame3_append_stage3_and_2x2_20260919_v1"
ATTRIBUTION_METRICS = (
    "smoke_iou_A001_A047",
    "selection_score_S_record_only",
    "three_class_miou_A001_A047_record_only",
    "fire_heat_iou_A001_A047_record_only",
    "fire_heat_iou_full134_record_only",
)


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_package():
    manifest = read_json(ROOT / "MANIFEST.json")
    for row in manifest["files"]:
        path = ROOT / row["relative"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Append Stage 3 package drift: {row['relative']}")
    if sha256(ROOT / "AUTHORIZATION.md") != manifest["authorization_sha256"]:
        raise RuntimeError("Authorization record changed")
    return sha256(ROOT / "MANIFEST.json")


def load_perturb():
    path = BUNDLE / "project_support/FLAME3_SUBMISSION_SOURCE_20260822/src/evaluate_test107_posthoc.py"
    if sha256(path) != PERTURB_SOURCE_SHA256:
        raise RuntimeError("Frozen evaluate_test107_posthoc.py changed")
    spec = importlib.util.spec_from_file_location("frozen_posthoc_perturb_append", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.perturb, path


def inventory():
    val_csv = BUNDLE / VAL_REL
    rows = load_csv(val_csv)
    if len(rows) != 134 or len({row["sample_key"] for row in rows}) != 134:
        raise RuntimeError("Frozen val134 inventory changed")
    if sum(row["sample_class"] == "No Fire" for row in rows) != 35:
        raise RuntimeError("Frozen No-Fire count changed")
    allowed = input_paths(val_csv, rows, BUNDLE)
    annotation = BUNDLE / ANNOTATION_REL
    manifest = annotation / "flame3_threeclass_annotation_manifest_150.json"
    freeze = annotation / "final_incremental_audit_20260815/ANNOTATION_FINAL_FREEZE.json"
    items = [item for item in read_json(manifest)["items"] if item["split"] == "val"]
    if sorted(item["annotation_id"] for item in items) != [f"A{i:03d}" for i in range(1, 48)]:
        raise RuntimeError("Frozen manual A001-A047 set changed")
    allowed |= {
        normalized(annotation / "completed_masks" / (item["annotation_id"] + ".png"))
        for item in items
    }
    hashed = allowed | {normalized(manifest), normalized(freeze)}
    previous = read_json(APPEND_STAGE2 / "PREFLIGHT.json")["input_sha256"]
    current = {path: stage2_sha(path) for path in sorted(hashed)}
    missing = [path for path in current if path not in previous]
    changed = [path for path, digest in current.items() if previous.get(path) != digest]
    if missing or changed:
        raise RuntimeError(f"Append Stage 3 input drift: missing={missing[:3]} changed={changed[:3]}")
    report_outputs = {
        normalized(ROOT / "APPEND_STAGE3_RESULTS.csv"),
        normalized(ROOT / "ATTRIBUTION_2X2.csv"),
    }
    return allowed | report_outputs, current


def checkpoint_units():
    units = []
    for arm in ARMS:
        for seed in SEEDS:
            folder = APPEND_STAGE2 / "runs" / arm / f"seed{seed}"
            result = read_json(folder / "RESULT.json")
            if result["status"] != "COMPLETE_30_EPOCHS" or not result["input_hashes_unchanged"]:
                raise RuntimeError(f"Incomplete append Stage 2 result: {arm} seed{seed}")
            expected_hashes = result["checkpoint_sha256"]
            for epoch in EPOCHS:
                path = folder / f"epoch{epoch}.pth"
                if sha256(path) != expected_hashes[path.name]:
                    raise RuntimeError(f"Append checkpoint drift: {path}")
                units.append({"arm": arm, "seed": seed, "epoch": epoch, "path": path})
    if len(units) != 45:
        raise RuntimeError(f"Expected 45 append checkpoint units, got {len(units)}")
    return units


def context_paths():
    paths = [FIRST_STAGE3 / "STAGE3_SUMMARY.json"]
    for seed in SEEDS:
        paths.append(FIRST_STAGE3 / "results/baseline_v11" / f"seed{seed}" / "epoch100.json")
        for epoch in EPOCHS:
            paths.append(FIRST_STAGE3 / "results/R1" / f"seed{seed}" / f"epoch{epoch}.json")
    return paths


class PerturbedLoader:
    def __init__(self, loader, trainer, perturb, condition, device):
        self.loader = loader
        self.trainer = trainer
        self.perturb = perturb
        self.condition = condition
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            items = list(batch)
            names = self.trainer.extract_flame3_sample_keys(items[3])
            images = items[0].to(device=self.device, dtype=torch.float, non_blocking=True)
            items[0] = self.perturb(images, names, self.condition)
            yield tuple(items)


def build_runtime(payload, arm, seed):
    rt, trainer, factory = runtime(APPEND_STAGE2 / "source")
    config = dict(payload["config"])
    config.update(
        ROOTDATASET=str(BUNDLE.resolve()),
        FLAME3_MANUAL_SMOKE_V2_ANNOTATION_PACKAGE=str((BUNDLE / ANNOTATION_REL).resolve()),
        DEVICE="cuda:0", BATCHSIZE=8, NUM_WORKERS=0,
    )
    dataset = rt.build_dataset(config, "val")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    baseline = rt.build_model(config, augment=True)
    model = apply_arm(baseline, arm, seed=seed)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.cuda().eval()
    criterion = make_criterion(rt, factory, config, arm)
    targets = trainer.load_frozen_manual_dev_targets(config["FLAME3_MANUAL_SMOKE_V2_ANNOTATION_PACKAGE"])
    return trainer, config, loader, model, criterion, targets


def expected_clean(unit):
    row = read_json(
        APPEND_STAGE2 / "runs" / unit["arm"] / f"seed{unit['seed']}"
        / "epochs" / f"epoch{unit['epoch']:02d}.json"
    )
    return metric_view(row["validation"])


def evaluate_unit(unit, perturb, output):
    payload = torch.load(unit["path"], map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != unit["epoch"]:
        raise RuntimeError(f"Checkpoint epoch mismatch: {unit['path']}")
    if int(payload.get("config", {}).get("SEED", -1)) != unit["seed"]:
        raise RuntimeError(f"Checkpoint seed mismatch: {unit['path']}")
    expected = expected_clean(unit)
    trainer, config, loader, model, criterion, targets = build_runtime(
        payload, unit["arm"], unit["seed"]
    )
    result = {
        "protocol": PROTOCOL_ID,
        "arm": unit["arm"], "seed": unit["seed"], "epoch": unit["epoch"],
        "checkpoint": str(unit["path"]), "checkpoint_sha256": sha256(unit["path"]),
        "conditions": {}, "training_performed": False, "test107_read": False,
        "predictions_saved": False,
    }
    device = torch.device("cuda:0")
    for condition in CONDITIONS:
        wrapped = PerturbedLoader(loader, trainer, perturb, condition, device)
        metrics, _, _, _ = trainer.run_validation(
            model, wrapped, criterion, config, unit["epoch"] - 1, None, True, targets
        )
        result["conditions"][condition["id"]] = metric_view(metrics)
    replay = result["conditions"]["clean"]
    result["clean_replay_max_abs"] = max(abs(replay[key] - expected[key]) for key in expected)
    result["clean_replay_contract_pass"] = result["clean_replay_max_abs"] < 1e-6
    if not result["clean_replay_contract_pass"]:
        raise RuntimeError(f"Clean replay drift: {unit}: {result['clean_replay_max_abs']}")
    assert_finite(result)
    write_json(output, result)
    del model, payload
    torch.cuda.empty_cache()
    return result


def robust_windows(root, arms):
    output = {arm: {} for arm in arms}
    for arm in arms:
        for seed in SEEDS:
            by_epoch = [
                read_json(root / "results" / arm / f"seed{seed}" / f"epoch{epoch}.json")
                for epoch in EPOCHS
            ]
            output[arm][str(seed)] = {
                condition["id"]: average_views(
                    [item["conditions"][condition["id"]] for item in by_epoch]
                )
                for condition in CONDITIONS
            }
    return output


def clean_windows(root, arms):
    output = {arm: {} for arm in arms}
    for arm in arms:
        for seed in SEEDS:
            output[arm][str(seed)] = clean_window_from_records(
                read_jsonl(root / "runs" / arm / f"seed{seed}" / "metrics.jsonl")
            )
    return output


def baseline_clean_windows():
    output = {}
    for seed in SEEDS:
        folder = (
            BUNDLE / "project_support/experiments/flame3_pidnet_s_fusion_manual_smoke_v11"
            / f"flame3_fusion_manual_smoke_v11_30e_seed{seed}"
        )
        output[str(seed)] = clean_window_from_records(read_jsonl(folder / "metrics.jsonl"))
    return output


def baseline_context_windows():
    return {
        str(seed): read_json(
            FIRST_STAGE3 / "results/baseline_v11" / f"seed{seed}" / "epoch100.json"
        )["conditions"]
        for seed in SEEDS
    }


def append_decisions(clean, robust, baseline_clean, baseline_context, efficiency):
    decisions = {}
    for arm in ARMS:
        seeds = []
        for seed in SEEDS:
            candidate = clean[arm][str(seed)]
            reference = baseline_clean[str(seed)]
            smoke_delta = candidate["smoke_iou_A001_A047"] - reference["smoke_iou_A001_A047"]
            fp_delta = (
                candidate["no_fire_joint_false_positive_ratio"]
                - reference["no_fire_joint_false_positive_ratio"]
            )
            candidate_drop = (
                robust[arm][str(seed)]["clean"]["smoke_iou_A001_A047"]
                - robust[arm][str(seed)]["thermal_noise_005"]["smoke_iou_A001_A047"]
            )
            baseline_drop = (
                baseline_context[str(seed)]["clean"]["smoke_iou_A001_A047"]
                - baseline_context[str(seed)]["thermal_noise_005"]["smoke_iou_A001_A047"]
            )
            seeds.append({
                "seed": seed,
                "clean_smoke_delta": smoke_delta,
                "no_fire_joint_fp_delta": fp_delta,
                "guards": {
                    "clean_smoke": candidate["smoke_iou_A001_A047"]
                    >= reference["smoke_iou_A001_A047"] - 0.010,
                    "no_fire_absolute": candidate["no_fire_joint_false_positive_ratio"] <= 0.005,
                    "no_fire_increase": fp_delta <= 0.001,
                },
                "precision_positive": smoke_delta >= 0.020,
                "candidate_sigma005_drop": candidate_drop,
                "baseline_epoch100_context_sigma005_drop": baseline_drop,
                "descriptive_robust_advantage_vs_baseline_epoch100_context": baseline_drop - candidate_drop,
                "robust_rule_eligible": False,
                "robust_rule_reason": "V1.1_EPOCH26_30_CHECKPOINTS_NOT_AVAILABLE",
            })
        guards = all(all(row["guards"].values()) for row in seeds)
        precision = all(row["precision_positive"] for row in seeds)
        decisions[arm] = {
            "seeds": seeds,
            "clean_guards_3_of_3": guards,
            "precision_rule_3_of_3": precision,
            "precision_axis_pass": guards and precision,
            "robust_axis_decision": "NOT_DETERMINABLE_BASELINE_WINDOW_MISSING",
            "efficiency": efficiency[arm],
            "efficiency_control_pass": False,
            "formal_latency_status": "DEFERRED_BY_AUTHORIZATION",
            "overall_decision": (
                "PASS_PRECISION_AXIS" if guards and precision
                else "NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES"
            ),
            "S_used_for_decision": False,
            "Fire_used_for_positive_decision": False,
        }
    return decisions


def attribution_2x2(baseline, r1, r2, r3):
    rows = []
    aggregate = {}
    condition_ids = [condition["id"] for condition in CONDITIONS if condition["id"] != "clean"]
    for condition in condition_ids:
        aggregate[condition] = {}
        for metric in ATTRIBUTION_METRICS:
            metric_rows = []
            for seed in SEEDS:
                b0 = float(baseline[str(seed)][condition][metric])
                y_r1 = float(r1[str(seed)][condition][metric])
                y_r2 = float(r2[str(seed)][condition][metric])
                y_r3 = float(r3[str(seed)][condition][metric])
                row = {
                    "condition": condition, "metric": metric, "seed": seed,
                    "Y_B0": b0, "Y_R1": y_r1, "Y_R2": y_r2, "Y_R3": y_r3,
                    "R1_minus_B0": y_r1 - b0,
                    "R2_minus_B0": y_r2 - b0,
                    "interaction": y_r3 - y_r1 - y_r2 + b0,
                    "Fire_used_for_decision": False,
                }
                rows.append(row)
                metric_rows.append(row)
            summary = {}
            for key in ("Y_B0", "Y_R1", "Y_R2", "Y_R3", "R1_minus_B0", "R2_minus_B0", "interaction"):
                values = [row[key] for row in metric_rows]
                summary[key] = {
                    "mean": statistics.fmean(values),
                    "sample_sd": statistics.stdev(values),
                    "by_seed": values,
                }
            aggregate[condition][metric] = summary
    return {
        "definition": {
            "B0": "v1.1 no reliability gate, no thermal degradation training",
            "R1": "thermal degradation training only",
            "R2": "reliability gate only",
            "R3": "reliability gate plus thermal degradation training",
            "interaction": "Y_R3 - Y_R1 - Y_R2 + Y_B0",
        },
        "formal_attribution_eligible": False,
        "limitation": (
            "B0 perturbation values use retained v1.1 epoch100 checkpoints; R1/R2/R3 use the frozen "
            "epoch26-30 window. Same-seed effects are descriptive context, not a formal symmetric-window attribution."
        ),
        "conditions": condition_ids,
        "metrics": list(ATTRIBUTION_METRICS),
        "aggregate": aggregate,
        "rows": rows,
        "Fire_used_for_decision": False,
    }


def aggregate_all(efficiency):
    append_clean = clean_windows(APPEND_STAGE2, ARMS)
    append_robust = robust_windows(ROOT, ARMS)
    baseline_clean = baseline_clean_windows()
    baseline_context = baseline_context_windows()
    r1_clean = clean_windows(FIRST_STAGE2, ("R1",))["R1"]
    r1_robust = robust_windows(FIRST_STAGE3, ("R1",))["R1"]
    decisions = append_decisions(
        append_clean, append_robust, baseline_clean, baseline_context, efficiency
    )
    attribution = attribution_2x2(
        baseline_context, r1_robust, append_robust["R2"], append_robust["R3"]
    )
    flat = []
    for arm in ARMS:
        for seed in SEEDS:
            clean = append_clean[arm][str(seed)]
            robust = append_robust[arm][str(seed)]
            row = {
                "arm": arm, "seed": seed,
                "clean_smoke_iou": clean["smoke_iou_A001_A047"],
                "clean_background_iou": clean["background_iou_A001_A047"],
                "clean_miou_record_only": clean["three_class_miou_A001_A047_record_only"],
                "clean_S_record_only": clean["selection_score_S_record_only"],
                "clean_fire_full134_record_only": clean["fire_heat_iou_full134_record_only"],
                "no_fire_joint_fp": clean["no_fire_joint_false_positive_ratio"],
                "deploy_parameters": efficiency[arm]["deploy_parameters"],
                "gmacs_proxy": efficiency[arm]["gmacs_proxy"],
                "gmacs_change_percent": efficiency[arm]["gmacs_change_percent"],
                "formal_latency_status": "DEFERRED_BY_AUTHORIZATION",
            }
            for condition in CONDITIONS[1:]:
                cid = condition["id"]
                row[cid + "_smoke_iou"] = robust[cid]["smoke_iou_A001_A047"]
                row[cid + "_S_record_only"] = robust[cid]["selection_score_S_record_only"]
                row[cid + "_Fire_record_only"] = robust[cid]["fire_heat_iou_A001_A047_record_only"]
            flat.append(row)
    summary = {
        "protocol": PROTOCOL_ID,
        "status": "COMPLETE_S1_R2_R3_9_RUNS_THREE_AXES_REPORTED",
        "baseline_clean_epoch26_30": baseline_clean,
        "baseline_robust_epoch100_context_only": baseline_context,
        "r1_clean_epoch26_30": r1_clean,
        "r1_robust_epoch26_30": r1_robust,
        "append_clean_epoch26_30": append_clean,
        "append_robust_epoch26_30": append_robust,
        "decisions": decisions,
        "efficiency_static": efficiency,
        "robust_axis_limitation": "V1.1_EPOCH26_30_CHECKPOINTS_NOT_AVAILABLE",
        "formal_latency_status": "DEFERRED_BY_AUTHORIZATION",
        "S_used_for_decision": False,
        "Fire_used_for_positive_decision": False,
        "test107_read": False,
        "training_performed": False,
    }
    assert_finite(summary)
    assert_finite(attribution)
    write_json(ROOT / "APPEND_STAGE3_SUMMARY.json", summary)
    write_csv(ROOT / "APPEND_STAGE3_RESULTS.csv", flat)
    write_json(ROOT / "ATTRIBUTION_2X2.json", attribution)
    write_csv(ROOT / "ATTRIBUTION_2X2.csv", attribution["rows"])
    return summary, attribution


def preflight():
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Approved RTX4090 is required")
    package_sha = verify_package()
    first_status = read_json(FIRST_STAGE3 / "STAGE3_STATUS.json")
    append_status = read_json(APPEND_STAGE2 / "S1_R2_R3_APPEND_STATUS.json")
    if first_status.get("status") != "COMPLETE" or not first_status.get("input_hashes_unchanged"):
        raise RuntimeError("First-batch Stage 3 is not complete and frozen")
    if append_status.get("status") != "COMPLETE_S1_R2_R3_9_RUNS" or append_status.get("failed"):
        raise RuntimeError("Append Stage 2 is not complete and clean")
    allowed, input_hashes = inventory()
    units = checkpoint_units()
    perturb, perturb_path = load_perturb()
    contexts = {str(path.resolve()): sha256(path) for path in context_paths()}
    install_guard(allowed)
    result = {
        "protocol": PROTOCOL_ID,
        "status": "PASS",
        "unit_count": len(units),
        "candidate_runs": 9,
        "candidate_checkpoints": 45,
        "conditions": list(CONDITIONS),
        "input_sha256": input_hashes,
        "context_sha256": contexts,
        "package_manifest_sha256": package_sha,
        "perturb_source": str(perturb_path),
        "perturb_source_sha256": PERTURB_SOURCE_SHA256,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "training_performed": False, "test107_read": False,
    }
    write_json(ROOT / "PREFLIGHT.json", result)
    return result, units, perturb, input_hashes, contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--probe-one", action="store_true")
    args = parser.parse_args()
    failure = ROOT / "APPEND_STAGE3_FAILURE.json"
    if failure.exists():
        failure.unlink()
    try:
        pref, units, perturb, before, contexts_before = preflight()
        if args.preflight_only:
            print(json.dumps({key: value for key, value in pref.items() if key not in ("input_sha256", "context_sha256")}, indent=2))
            return
        selected = units[:1] if args.probe_one else units
        for index, unit in enumerate(selected, start=1):
            output = ROOT / "results" / unit["arm"] / f"seed{unit['seed']}" / f"epoch{unit['epoch']}.json"
            write_json(ROOT / "APPEND_STAGE3_STATUS.json", {
                "status": "RUNNING_PROBE" if args.probe_one else "RUNNING",
                "unit_index": index, "unit_total": len(selected),
                "current": {key: str(value) if isinstance(value, Path) else value for key, value in unit.items()},
                "updated_unix": time.time(), "test107_read": False,
            })
            if output.exists():
                existing = read_json(output)
                if existing["checkpoint_sha256"] != sha256(unit["path"]):
                    raise RuntimeError(f"Existing result/checkpoint mismatch: {output}")
                continue
            evaluate_unit(unit, perturb, output)
        if args.probe_one:
            write_json(ROOT / "PROBE_STATUS.json", {
                "status": "PASS",
                "unit": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in selected[0].items()
                },
                "test107_read": False,
            })
            return
        efficiency = read_json(ROOT / "EFFICIENCY_STATIC.json")
        summary, attribution = aggregate_all(efficiency)
        after = {path: stage2_sha(path) for path in before}
        contexts_after = {str(path.resolve()): sha256(path) for path in context_paths()}
        if after != before or contexts_after != contexts_before:
            raise RuntimeError("Read-only inputs or first-batch context changed during append Stage 3")
        write_json(ROOT / "APPEND_STAGE3_STATUS.json", {
            "status": "COMPLETE", "unit_total": len(units),
            "summary_status": summary["status"],
            "attribution_rows": len(attribution["rows"]),
            "finished_unix": time.time(),
            "input_hashes_unchanged": True, "context_hashes_unchanged": True,
            "test107_read": False, "training_performed": False,
        })
    except BaseException as exc:
        write_json(failure, {
            "status": "FAILED_STOP", "error": repr(exc),
            "traceback": traceback.format_exc(),
            "test107_read": False, "training_performed": False,
        })
        raise


if __name__ == "__main__":
    main()
