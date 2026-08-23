from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


FINAL_DATE = "2026-08-22"
EXPECTED_SPLIT_HASHES = {
    "train": "108a73f11490b342160804838b54591667fbdb089c25d3aa90668990a14122e3",
    "val": "c3899b01604b9c3d62cf7f6be4370d12b6f794151017bbcad27057851830b0e1",
    "test107": "3ebdccd489d4e71ce87c0bd6f7b75faf34cc051b89d1da137da1ac0f6b7b9355",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the frozen FLAME3 submission package.")
    parser.add_argument(
        "--submission-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    root = args.submission_root.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "passed" if condition else "failed",
                "evidence": evidence,
            }
        )

    critical_paths = {
        "flame2_result": root / "results" / "flame2_zero_shot" / "FLAME2_ZERO_SHOT_COMPLETE.json",
        "flame2_report": root / "results" / "flame2_zero_shot" / "FLAME2_ZERO_SHOT_REPORT.md",
        "posthoc_result": root / "results" / "test107_posthoc" / "POSTHOC_COMPLETE.json",
        "posthoc_report": root / "results" / "test107_posthoc" / "POSTHOC_REPORT.md",
        "conv1_result": root / "results" / "conv1_initialization_ablation_30e" / "CONV1_INITIALIZATION_ABLATION_COMPLETE.json",
        "conv1_report": root / "results" / "conv1_initialization_ablation_30e" / "CONV1_INITIALIZATION_ABLATION_REPORT.md",
        "efficiency": root / "results" / "efficiency" / "RTX4090_random_stem_seed200_epoch30.json",
        "complete_file_manifest": root / "reproducibility" / "complete_split_file_manifest" / "FLAME3_COMPLETE_SPLIT_AND_FILE_HASH_MANIFEST.json",
        "complete_file_manifest_csv": root / "reproducibility" / "complete_split_file_manifest" / "FLAME3_COMPLETE_SPLIT_AND_FILE_HASH_MANIFEST.csv",
        "environment": root / "environment.yml",
        "requirements": root / "requirements-lock.txt",
        "license": root / "reproducibility" / "DATA_LICENSE.md",
    }
    missing_critical = [name for name, path in critical_paths.items() if not path.is_file()]
    check("critical_artifacts_exist", not missing_critical, missing_critical)
    if missing_critical:
        raise FileNotFoundError(f"Missing critical artifacts: {missing_critical}")

    split_paths = {
        "train": root / "reproducibility" / "splits" / "train.csv",
        "val": root / "reproducibility" / "splits" / "val.csv",
        "test107": root / "reproducibility" / "splits" / "test107_manual_threeclass.csv",
    }
    split_counts = {name: len(csv_rows(path)) for name, path in split_paths.items()}
    split_hashes = {name: sha256_file(path) for name, path in split_paths.items()}
    check("split_row_counts", split_counts == {"train": 493, "val": 134, "test107": 107}, split_counts)
    check("split_sha256", split_hashes == EXPECTED_SPLIT_HASHES, split_hashes)

    hash_index_path = root / "reproducibility" / "REPRODUCIBILITY_FILE_HASHES.json"
    hash_index = load_json(hash_index_path)
    hash_index_mismatches: list[dict[str, Any]] = []
    for record in hash_index:
        path = root / "reproducibility" / str(record["path"])
        actual = {
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        if actual["sha256"] != record["sha256"] or actual["bytes"] != record["bytes"]:
            hash_index_mismatches.append(
                {"path": record["path"], "expected": record, "actual": actual}
            )
    check("reproducibility_hash_index", not hash_index_mismatches, hash_index_mismatches)

    complete_manifest = load_json(critical_paths["complete_file_manifest"])
    complete_manifest_rows = csv_rows(critical_paths["complete_file_manifest_csv"])
    complete_manifest_ok = (
        complete_manifest.get("status") == "complete"
        and complete_manifest.get("split_counts") == split_counts
        and complete_manifest.get("unique_sample_count") == 734
        and complete_manifest.get("source_csv_sha256") == split_hashes
        and complete_manifest.get("training_or_inference_performed") is False
        and len(complete_manifest_rows) == 734
        and sha256_file(critical_paths["complete_file_manifest_csv"])
        == complete_manifest.get("manifest_csv_sha256")
    )
    check("complete_per_file_manifest", complete_manifest_ok, complete_manifest)

    flame2 = load_json(critical_paths["flame2_result"])
    flame2_arms = [str(run.get("arm")) for run in flame2.get("runs", [])]
    flame2_ok = (
        flame2.get("status") == "complete"
        and flame2.get("preflight", {}).get("source_test_rows") == 200
        and flame2_arms.count("baseline") == 3
        and flame2_arms.count("v11") == 3
        and flame2.get("training_performed") is False
        and flame2.get("threshold_tuning_performed") is False
        and flame2.get("predictions_saved") is False
        and flame2.get("flame3_test_read") is False
    )
    check("flame2_zero_shot_complete", flame2_ok, {"run_arms": flame2_arms, "preflight": flame2.get("preflight", {})})
    flame2_per_image_rows = len(csv_rows(root / "results" / "flame2_zero_shot" / "PER_IMAGE_METRICS.csv"))
    check("flame2_per_image_metrics", flame2_per_image_rows == 1200, flame2_per_image_rows)

    posthoc = load_json(critical_paths["posthoc_result"])
    posthoc_ok = (
        posthoc.get("status") == "complete"
        and posthoc.get("run_count") == 6
        and posthoc.get("perturbation_count") == 29
        and posthoc.get("per_image_confusion_rows") == 18618
        and posthoc.get("strata_rows") == 90
        and posthoc.get("strata_aggregate_rows") == 30
        and posthoc.get("strata_paired_rows") == 15
        and all(item.get("clean_confusion_exactly_matches_frozen_result") is True for item in posthoc.get("clean_equivalence", []))
        and posthoc.get("training_performed") is False
        and posthoc.get("threshold_tuning_performed") is False
        and posthoc.get("predictions_saved") is False
        and posthoc.get("method_identity_changed") is False
    )
    check("test107_posthoc_complete", posthoc_ok, posthoc)

    conv1 = load_json(critical_paths["conv1_result"])
    conv1_status = load_json(root / "artifacts" / "conv1_initialization_ablation" / "STATUS.json")
    conv1_ok = (
        conv1.get("status") == "complete"
        and conv1.get("decision") == "retain_random_four_channel_stem"
        and conv1.get("guards_clear") is False
        and conv1.get("positive_seed_count_primary") == 2
        and conv1.get("test_split_read") is False
        and conv1.get("training_protocol_modified_after_preregistration") is False
        and conv1_status.get("state") == "complete"
        and conv1_status.get("test_split_read") is False
    )
    check("conv1_ablation_frozen_decision", conv1_ok, {"result": conv1, "status": {"state": conv1_status.get("state"), "test_split_read": conv1_status.get("test_split_read")}})

    run_root = root / "artifacts" / "conv1_initialization_ablation" / "runs"
    expected_runs = [f"arm{arm}_seed{seed}" for arm in ("A", "B") for seed in (200, 201, 202)]
    run_audit: dict[str, Any] = {}
    run_metadata_ok = True
    for name in expected_runs:
        directory = run_root / name
        expected_files = {"environment.json", "metrics.jsonl", "resolved_config.json", "run_summary.json"}
        actual_files = {path.name for path in directory.glob("*") if path.is_file()}
        records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (directory / "metrics.jsonl").is_file() else []
        epochs = [int(record["epoch"]) for record in records]
        item_ok = actual_files == expected_files and epochs == list(range(1, 31))
        run_metadata_ok = run_metadata_ok and item_ok
        run_audit[name] = {"files": sorted(actual_files), "epoch_count": len(epochs), "epochs": epochs, "status": "passed" if item_ok else "failed"}
    check("conv1_six_run_metadata", run_metadata_ok, run_audit)

    efficiency = load_json(critical_paths["efficiency"])
    inference = efficiency.get("inference", {})
    latency = efficiency.get("latency", {})
    complexity = efficiency.get("complexity", {})
    efficiency_ok = (
        efficiency.get("status") == "complete"
        and inference.get("augment") is False
        and inference.get("batch_size") == 1
        and inference.get("input_height") == 512
        and inference.get("input_width") == 640
        and inference.get("amp") is True
        and latency.get("warmup_iterations") == 100
        and latency.get("timed_iterations_per_trial") == 200
        and latency.get("trials") == 10
        and latency.get("sample_count") == 2000
        and all(key in latency for key in ("mean_ms", "median_ms", "p95_ms", "fps_from_mean", "peak_allocated_vram_mb", "peak_reserved_vram_mb"))
        and all(key in complexity for key in ("parameters", "macs", "flops_two_per_mac"))
        and efficiency.get("data_loading") is not None
        and efficiency.get("test_split_read") is False
        and efficiency.get("training_performed") is False
    )
    check("efficiency_protocol_complete", efficiency_ok, efficiency)

    python_files = list(root.rglob("*.py"))
    sys_path_hits: list[str] = []
    sys_path_insert_token = "sys.path" + ".insert"
    for path in python_files:
        if sys_path_insert_token in path.read_text(encoding="utf-8", errors="replace"):
            sys_path_hits.append(path.relative_to(root).as_posix())
    check("no_sys_path_insert", not sys_path_hits, sys_path_hits)

    machine_path_hits: list[dict[str, Any]] = []
    absolute_windows_path = re.compile(r"[A-Za-z]:" + re.escape(chr(92)))
    for directory in (root / "src", root / "configs"):
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json"}:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if absolute_windows_path.search(line):
                    machine_path_hits.append({"path": path.relative_to(root).as_posix(), "line": number})
    check("no_machine_paths_in_src_or_configs", not machine_path_hits, machine_path_hits)

    pidnet_path = root / "third_party" / "RoboFireFuseNet" / "models" / "pidnet.py"
    pidnet_lines = pidnet_path.read_text(encoding="utf-8").splitlines()
    summary_lines = [index for index, line in enumerate(pidnet_lines) if "from torchsummary import summary" in line]
    summary_guarded = (
        len(summary_lines) == 1
        and any("if __name__ == '__main__':" in line or 'if __name__ == "__main__":' in line for line in pidnet_lines[max(0, summary_lines[0] - 5):summary_lines[0]])
    )
    check("torchsummary_import_main_only", summary_guarded, summary_lines)

    forbidden_binaries = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
    ]
    generated_caches = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name == "__pycache__" or path.suffix.lower() in {".pyc", ".pyo"}
    ]
    check("no_checkpoint_binaries", not forbidden_binaries, forbidden_binaries)
    check("no_python_caches", not generated_caches, generated_caches)

    critical_hashes = {
        name: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in critical_paths.items()
    }
    failed = [item["name"] for item in checks if item["status"] != "passed"]
    result = {
        "protocol": "flame3_submission_package_completion_audit_v1",
        "created_on": FINAL_DATE,
        "status": "complete" if not failed else "failed",
        "failed_checks": failed,
        "checks": checks,
        "critical_artifact_hashes": critical_hashes,
        "frozen_conclusions": {
            "conv1_initialization": conv1.get("decision"),
            "conv1_mean_delta_manual_three_class_miou": conv1.get("mean_paired_deltas", {}).get("manual_A001_A047_three_class_mIoU"),
            "conv1_mean_delta_fire_heat_iou": conv1.get("mean_paired_deltas", {}).get("full_val134_fire_heat_iou"),
            "efficiency_fps": latency.get("fps_from_mean"),
            "efficiency_mean_ms": latency.get("mean_ms"),
            "efficiency_p95_ms": latency.get("p95_ms"),
            "parameter_count": complexity.get("parameters"),
            "gmacs": complexity.get("gmacs"),
            "gflops_two_per_mac": complexity.get("gflops_two_per_mac"),
        },
    }
    output = root / "reproducibility" / "SUBMISSION_COMPLETION_AUDIT.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# FLAME3 submission readiness audit",
        "",
        f"Status: `{result['status']}`. Failed checks: {', '.join(failed) if failed else 'none'}.",
        "",
        "## Frozen conclusions",
        "",
        f"- Conv1 initialization: `{conv1.get('decision')}`. The RGB-copy/IR-mean arm improved manual three-class mIoU by {conv1.get('mean_paired_deltas', {}).get('manual_A001_A047_three_class_mIoU'):+.6f}, but Fire/Heat IoU changed by {conv1.get('mean_paired_deltas', {}).get('full_val134_fire_heat_iou'):+.6f} and triggered the preregistered guard.",
        f"- RTX 4090 efficiency: {latency.get('mean_ms'):.3f} ms mean, {latency.get('p95_ms'):.3f} ms P95, {latency.get('fps_from_mean'):.2f} FPS; {complexity.get('parameters'):,} parameters, {complexity.get('gmacs'):.3f} GMACs.",
        f"- Complete data manifest: {complete_manifest.get('unique_sample_count')} unique samples across train/val/test107; hashing performed without training or inference.",
        "- FLAME2 is reported as a coverage-mismatched zero-shot lower-bound transfer estimate. test107 strata/perturbations remain explicitly post-hoc and are not method-selection evidence.",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    report.extend(f"| {item['name']} | {item['status']} |" for item in checks)
    (root / "SUBMISSION_READINESS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed_checks": failed}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
