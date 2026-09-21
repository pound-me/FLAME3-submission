"""Read-only stage-3 evaluator for the approved 18 FLAME3 structure runs."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent
STAGE2 = BUNDLE / "stage2_structure_20260918_v1"
sys.path.insert(0, str(BUNDLE / "project_support_night_20260824/.deps"))
sys.path.insert(0, str(STAGE2 / "implementation"))
sys.path.insert(0, str(STAGE2))

import torch
from torch.utils.data import DataLoader

from engineering_checks import make_criterion, runtime
from structure_arms import apply_arm
from stage2_protocol import ANNOTATION_REL, VAL_REL, input_paths, install_guard, normalized, sha as stage2_sha
from stage3_protocol import (
    ARMS, CONDITIONS, EPOCHS, PERTURB_SOURCE_SHA256, PROTOCOL_ID, SEEDS,
    SUPPLEMENT_SHA256, assert_finite, average_views, clean_window_from_records,
    metric_view, read_json, read_jsonl, sha256, summarize_decisions, write_csv, write_json,
)


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_perturb():
    path = BUNDLE / "project_support/FLAME3_SUBMISSION_SOURCE_20260822/src/evaluate_test107_posthoc.py"
    if sha256(path) != PERTURB_SOURCE_SHA256:
        raise RuntimeError("Frozen evaluate_test107_posthoc.py changed")
    spec = importlib.util.spec_from_file_location("frozen_posthoc_perturb", path)
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
    previous = read_json(STAGE2 / "PREFLIGHT.json")["input_sha256"]
    current = {path: stage2_sha(path) for path in sorted(hashed)}
    missing = [path for path in current if path not in previous]
    changed = [path for path, digest in current.items() if previous.get(path) != digest]
    if missing or changed:
        raise RuntimeError(f"Stage3 input drift: missing={missing[:3]} changed={changed[:3]}")
    report_outputs = {normalized(ROOT / "STAGE3_RESULTS.csv")}
    return allowed | report_outputs, current


def checkpoint_units():
    units = []
    for seed in SEEDS:
        path = (
            BUNDLE / "project_support/experiments/flame3_pidnet_s_fusion_manual_smoke_v11"
            / f"flame3_fusion_manual_smoke_v11_30e_seed{seed}/last.pth"
        )
        units.append({"kind": "baseline_v11_context", "arm": "baseline_v11", "seed": seed, "epoch": 100, "path": path})
    for arm in ARMS:
        for seed in SEEDS:
            folder = STAGE2 / "runs" / arm / f"seed{seed}"
            result = read_json(folder / "RESULT.json")
            if result["status"] != "COMPLETE_30_EPOCHS" or not result["input_hashes_unchanged"]:
                raise RuntimeError(f"Incomplete stage2 result: {arm} seed{seed}")
            expected_hashes = result["checkpoint_sha256"]
            for epoch in EPOCHS:
                path = folder / f"epoch{epoch}.pth"
                if sha256(path) != expected_hashes[path.name]:
                    raise RuntimeError(f"Checkpoint drift: {path}")
                units.append({"kind": "candidate", "arm": arm, "seed": seed, "epoch": epoch, "path": path})
    return units


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
    rt, trainer, factory = runtime(STAGE2 / "source")
    config = dict(payload["config"])
    config.update(
        ROOTDATASET=str(BUNDLE.resolve()),
        FLAME3_MANUAL_SMOKE_V2_ANNOTATION_PACKAGE=str((BUNDLE / ANNOTATION_REL).resolve()),
        DEVICE="cuda:0", BATCHSIZE=8, NUM_WORKERS=0,
    )
    dataset = rt.build_dataset(config, "val")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    baseline = rt.build_model(config, augment=True)
    model = baseline if arm == "baseline_v11" else apply_arm(baseline, arm, seed=seed)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.cuda().eval()
    criterion = make_criterion(rt, factory, config, "baseline" if arm == "baseline_v11" else arm)
    targets = trainer.load_frozen_manual_dev_targets(config["FLAME3_MANUAL_SMOKE_V2_ANNOTATION_PACKAGE"])
    return trainer, config, loader, model, criterion, targets


def expected_clean(unit):
    if unit["kind"] == "candidate":
        row = read_json(
            STAGE2 / "runs" / unit["arm"] / f"seed{unit['seed']}"
            / "epochs" / f"epoch{unit['epoch']:02d}.json"
        )
        return metric_view(row["validation"])
    folder = (
        BUNDLE / "project_support/experiments/flame3_pidnet_s_fusion_manual_smoke_v11"
        / f"flame3_fusion_manual_smoke_v11_30e_seed{unit['seed']}"
    )
    row = next(item for item in read_jsonl(folder / "metrics.jsonl") if int(item["epoch"]) == 100)
    return metric_view(row["validation"])


def evaluate_unit(unit, perturb, output):
    payload = torch.load(unit["path"], map_location="cpu", weights_only=False)
    header_epoch = int(payload.get("epoch", -1))
    if unit["kind"] == "candidate" and header_epoch != unit["epoch"]:
        raise RuntimeError(f"Checkpoint epoch mismatch: {unit['path']}")
    if int(payload.get("config", {}).get("SEED", -1)) != unit["seed"]:
        raise RuntimeError(f"Checkpoint seed mismatch: {unit['path']}")
    expected = expected_clean(unit)
    legacy_endpoint_proof = None
    if unit["kind"] == "baseline_v11_context":
        embedded = metric_view(payload["validation_metrics"])
        maximum = max(abs(embedded[key] - expected[key]) for key in expected)
        if maximum != 0.0:
            raise RuntimeError(f"Legacy last.pth is not the historical epoch100 endpoint: {maximum}")
        legacy_endpoint_proof = {
            "checkpoint_header_epoch": header_epoch,
            "embedded_validation_equals_metrics_jsonl_epoch100": True,
            "maximum_abs_difference": maximum,
        }
    trainer, config, loader, model, criterion, targets = build_runtime(payload, unit["arm"], unit["seed"])
    result = {
        "protocol": PROTOCOL_ID, "kind": unit["kind"], "arm": unit["arm"],
        "seed": unit["seed"], "epoch": unit["epoch"], "checkpoint": str(unit["path"]),
        "checkpoint_sha256": sha256(unit["path"]), "conditions": {},
        "checkpoint_header_epoch": header_epoch,
        "legacy_epoch100_endpoint_proof": legacy_endpoint_proof,
        "training_performed": False, "test107_read": False, "predictions_saved": False,
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
    result["clean_replay_reference"] = (
        "same_checkpoint_stage2_epoch_record" if unit["kind"] == "candidate"
        else "v1.1_epoch100_context_historical_metric_different_torch_environment"
    )
    result["clean_replay_contract_pass"] = (
        result["clean_replay_max_abs"] < 1e-6 if unit["kind"] == "candidate" else None
    )
    assert_finite(result)
    write_json(output, result)
    del model, payload
    torch.cuda.empty_cache()
    return result


def aggregate(efficiency):
    baseline_clean, baseline_robust = {}, {}
    for seed in SEEDS:
        folder = (
            BUNDLE / "project_support/experiments/flame3_pidnet_s_fusion_manual_smoke_v11"
            / f"flame3_fusion_manual_smoke_v11_30e_seed{seed}"
        )
        baseline_clean[str(seed)] = clean_window_from_records(read_jsonl(folder / "metrics.jsonl"))
        baseline_robust[str(seed)] = read_json(
            ROOT / "results/baseline_v11" / f"seed{seed}" / "epoch100.json"
        )["conditions"]
    clean_windows = {arm: {} for arm in ARMS}
    robust_windows = {arm: {} for arm in ARMS}
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            run = STAGE2 / "runs" / arm / f"seed{seed}"
            clean_windows[arm][str(seed)] = clean_window_from_records(read_jsonl(run / "metrics.jsonl"))
            by_epoch = [
                read_json(ROOT / "results" / arm / f"seed{seed}" / f"epoch{epoch}.json")
                for epoch in EPOCHS
            ]
            robust_windows[arm][str(seed)] = {
                condition["id"]: average_views(
                    [item["conditions"][condition["id"]] for item in by_epoch]
                )
                for condition in CONDITIONS
            }
            clean = clean_windows[arm][str(seed)]
            robust = robust_windows[arm][str(seed)]
            flat = {
                "arm": arm, "seed": seed,
                "clean_smoke_iou": clean["smoke_iou_A001_A047"],
                "clean_background_iou": clean["background_iou_A001_A047"],
                "clean_miou_record_only": clean["three_class_miou_A001_A047_record_only"],
                "clean_S_record_only": clean["selection_score_S_record_only"],
                "clean_fire_full134_record_only": clean["fire_heat_iou_full134_record_only"],
                "no_fire_joint_fp": clean["no_fire_joint_false_positive_ratio"],
            }
            for condition in CONDITIONS[1:]:
                cid = condition["id"]
                flat[cid + "_smoke_iou"] = robust[cid]["smoke_iou_A001_A047"]
                flat[cid + "_S_record_only"] = robust[cid]["selection_score_S_record_only"]
            flat.update(
                deploy_parameters=efficiency[arm]["deploy_parameters"],
                gmacs_proxy=efficiency[arm]["gmacs_proxy"],
                gmacs_change_percent=efficiency[arm]["gmacs_change_percent"],
                formal_latency_status="DEFERRED_BY_AUTHORIZATION",
            )
            rows.append(flat)
    decisions = summarize_decisions(clean_windows, robust_windows, baseline_clean, baseline_robust, efficiency)
    summary = {
        "protocol": PROTOCOL_ID, "status": "COMPLETE_18_RUNS_THREE_AXES_REPORTED",
        "baseline_clean_epoch26_30": baseline_clean,
        "baseline_robust_epoch100_context_only": baseline_robust,
        "baseline_robust_window_limitation": (
            "V1.1 epoch26-30 checkpoints were not retained; epoch100 is context-only and the robust positive rule is not "
            "determinable without changing the approved symmetric window."
        ),
        "clean_windows": clean_windows, "robust_windows_epoch26_30": robust_windows,
        "decisions": decisions, "efficiency_static": efficiency,
        "formal_latency_status": "DEFERRED_BY_AUTHORIZATION",
        "S_used_for_decision": False, "Fire_used_for_positive_decision": False,
        "test107_read": False, "training_performed": False,
    }
    assert_finite(summary)
    write_json(ROOT / "STAGE3_SUMMARY.json", summary)
    write_csv(ROOT / "STAGE3_RESULTS.csv", rows)
    return summary


def preflight():
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Approved RTX4090 is required")
    manifest = read_json(ROOT / "MANIFEST.json")
    for row in manifest["files"]:
        if sha256(ROOT / row["relative"]) != row["sha256"]:
            raise RuntimeError(f"Stage3 package drift: {row['relative']}")
    if sha256(ROOT / "AUTHORIZATION.md") != manifest["authorization_sha256"]:
        raise RuntimeError("Authorization record changed")
    allowed, input_hashes = inventory()
    units = checkpoint_units()
    perturb, perturb_path = load_perturb()
    if len(units) != 93:
        raise RuntimeError(f"Expected 93 checkpoint units, got {len(units)}")
    install_guard(allowed)
    result = {
        "protocol": PROTOCOL_ID, "status": "PASS", "unit_count": len(units),
        "candidate_runs": 18, "candidate_checkpoints": 90, "baseline_context_epoch100_checkpoints": 3,
        "conditions": list(CONDITIONS), "input_sha256": input_hashes,
        "perturb_source": str(perturb_path), "perturb_source_sha256": PERTURB_SOURCE_SHA256,
        "supplement_sha256": SUPPLEMENT_SHA256, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "training_performed": False, "test107_read": False,
    }
    write_json(ROOT / "PREFLIGHT.json", result)
    return result, units, perturb, input_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--probe-one", action="store_true")
    args = parser.parse_args()
    failure_path = ROOT / "STAGE3_FAILURE.json"
    if failure_path.exists():
        failure_path.unlink()
    try:
        pref, units, perturb, before = preflight()
        if args.preflight_only:
            print(json.dumps({key: value for key, value in pref.items() if key != "input_sha256"}, indent=2))
            return
        selected = units[:1] if args.probe_one else units
        for index, unit in enumerate(selected, start=1):
            output = ROOT / "results" / unit["arm"] / f"seed{unit['seed']}" / f"epoch{unit['epoch']}.json"
            write_json(ROOT / "STAGE3_STATUS.json", {
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
                "status": "PASS", "unit": {
                    key: str(value) if isinstance(value, Path) else value for key, value in selected[0].items()
                }, "test107_read": False,
            })
            return
        efficiency = read_json(ROOT / "EFFICIENCY_STATIC.json")
        summary = aggregate(efficiency)
        after = {path: stage2_sha(path) for path in before}
        if after != before:
            raise RuntimeError("Read-only validation inputs changed during stage3")
        write_json(ROOT / "STAGE3_STATUS.json", {
            "status": "COMPLETE", "unit_total": len(units),
            "summary_status": summary["status"], "finished_unix": time.time(),
            "input_hashes_unchanged": True, "test107_read": False, "training_performed": False,
        })
    except BaseException as exc:
        write_json(failure_path, {
            "status": "FAILED_STOP", "error": repr(exc), "traceback": traceback.format_exc(),
            "test107_read": False, "training_performed": False,
        })
        raise


if __name__ == "__main__":
    main()
