from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .baseline_runtime import build_model, load_pretrained_if_available, seed_everything


SEEDS = (200, 201, 202)
RUN_ORDER = ((200, "A"), (200, "B"), (201, "B"), (201, "A"), (202, "A"), (202, "B"))
ARM_CONFIGS = {
    "A": "pidnet_s_fusion_manual_smoke_v11_30e.yaml",
    "B": "pidnet_s_fusion_manual_smoke_v11_conv1_rgb_irmean_30e.yaml",
}
ARM_RUN_PREFIXES = {
    "A": "conv1_random_30e",
    "B": "conv1_rgb_irmean_30e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen FLAME3 conv1 initialization ablation.")
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--annotation-package", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def config_path(root: Path, arm: str) -> Path:
    return root / "configs" / ARM_CONFIGS[arm]


def run_name(arm: str, seed: int) -> str:
    return f"{ARM_RUN_PREFIXES[arm]}_seed{seed}"


def run_directory(root: Path, config: dict[str, Any], arm: str, seed: int) -> Path:
    return root / "experiments" / str(config["EXPERIMENT_GROUP"]) / run_name(arm, seed)


def metric_epochs(path: Path) -> list[int]:
    if not path.is_file():
        return []
    return [
        int(json.loads(line)["epoch"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def config_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        key: [a.get(key), b.get(key)]
        for key in sorted(set(a).union(b))
        if a.get(key) != b.get(key)
    }


def initialization_fairness_audit(
    root: Path,
    pretrained: Path,
    configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    states: dict[str, dict[str, torch.Tensor]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for arm in ("A", "B"):
        seed_everything(200)
        model = build_model(configs[arm], augment=True)
        runtime_config = dict(configs[arm])
        runtime_config["PRETRAINED"] = str(pretrained)
        load_pretrained_if_available(model, runtime_config)
        states[arm] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        audits[arm] = runtime_config["PRETRAIN_LOAD_AUDIT"]

    stem_key = "conv1.0.weight"
    non_stem_mismatches = [
        key
        for key in states["A"]
        if key != stem_key and not torch.equal(states["A"][key], states["B"][key])
    ]
    source_payload = torch.load(pretrained, map_location="cpu", weights_only=True)
    source_state = source_payload.get("state_dict", source_payload.get("model_state_dict", source_payload))
    source_stem = source_state[stem_key]
    b_stem = states["B"][stem_key]
    rgb_exact = torch.equal(b_stem[:, :3], source_stem)
    ir_mean_exact = torch.equal(b_stem[:, 3:4], source_stem.mean(dim=1, keepdim=True))
    stems_differ = not torch.equal(states["A"][stem_key], states["B"][stem_key])
    passed = not non_stem_mismatches and rgb_exact and ir_mean_exact and stems_differ
    if not passed:
        raise RuntimeError(
            "Conv1 fairness audit failed: "
            f"non_stem_mismatches={non_stem_mismatches}, rgb_exact={rgb_exact}, "
            f"ir_mean_exact={ir_mean_exact}, stems_differ={stems_differ}"
        )
    return {
        "status": "passed",
        "seed": 200,
        "non_stem_mismatch_keys": non_stem_mismatches,
        "rgb_channels_equal_imagenet": rgb_exact,
        "ir_channel_equal_imagenet_rgb_mean": ir_mean_exact,
        "arm_stems_differ": stems_differ,
        "stem_sha256": {
            "A_random": tensor_sha256(states["A"][stem_key]),
            "B_rgb_irmean": tensor_sha256(states["B"][stem_key]),
            "source_imagenet_rgb": tensor_sha256(source_stem),
        },
        "pretrained_loading": audits,
    }


def preflight(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = args.submission_root.resolve()
    prereg_path = root / "reproducibility" / "CONV1_INITIALIZATION_ABLATION_PREREGISTRATION.json"
    required = [
        root / "src" / "train_baseline_v11.py",
        root / "src" / "summarize_conv1_ablation_30e.py",
        prereg_path,
        args.bundle_root.resolve(),
        args.pretrained.resolve(),
        args.train_csv.resolve(),
        args.val_csv.resolve(),
        args.annotation_package.resolve(),
    ] + [config_path(root, arm) for arm in ("A", "B")]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if "test" in args.train_csv.name.lower() or "test" in args.val_csv.name.lower():
        raise RuntimeError("Test split is forbidden")
    if read_csv_count(args.train_csv) != 493 or read_csv_count(args.val_csv) != 134:
        raise RuntimeError("Frozen train/validation row counts are not 493/134")
    prereg = load_json(prereg_path)
    if prereg.get("status") != "frozen_before_training":
        raise RuntimeError("Conv1 preregistration is not frozen")
    configs = {arm: load_yaml(config_path(root, arm)) for arm in ("A", "B")}
    allowed = set(prereg["allowed_config_differences"])
    drift = config_diff(configs["A"], configs["B"])
    if set(drift) != allowed:
        raise RuntimeError(f"Unexpected config differences: {drift}")
    for arm in ("A", "B"):
        expected_hash = prereg["arms"][arm]["config_sha256"]
        actual_hash = sha256_file(config_path(root, arm))
        if actual_hash != expected_hash:
            raise RuntimeError(f"Config {arm} hash drift: {actual_hash} != {expected_hash}")
    if configs["A"].get("PRETRAIN_STEM_STRATEGY") != "random":
        raise RuntimeError("Arm A strategy drift")
    if configs["B"].get("PRETRAIN_STEM_STRATEGY") != "rgb_copy_ir_mean":
        raise RuntimeError("Arm B strategy drift")
    fairness = initialization_fairness_audit(root, args.pretrained.resolve(), configs)
    payload = {
        "protocol": "flame3_conv1_initialization_ablation_preflight",
        "status": "passed",
        "preregistration": str(prereg_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "config_differences": drift,
        "config_sha256": {arm: sha256_file(config_path(root, arm)) for arm in ("A", "B")},
        "pretrained_sha256": sha256_file(args.pretrained.resolve()),
        "train_csv_sha256": sha256_file(args.train_csv.resolve()),
        "val_csv_sha256": sha256_file(args.val_csv.resolve()),
        "train_rows": 493,
        "val_rows": 134,
        "initialization_fairness": fairness,
        "test_split_read": False,
    }
    out = root / "artifacts" / "conv1_initialization_ablation" / "PREFLIGHT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return configs, payload


def training_command(
    args: argparse.Namespace,
    root: Path,
    arm: str,
    seed: int,
    resume: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train_baseline_v11",
        "--config",
        str(config_path(root, arm)),
        "--root-dataset",
        str(args.bundle_root.resolve()),
        "--pretrained",
        str(args.pretrained.resolve()),
        "--trainset",
        str(args.train_csv.resolve()),
        "--validset",
        str(args.val_csv.resolve()),
        "--manual-smoke-annotation-package",
        str(args.annotation_package.resolve()),
        "--batch-size",
        "8",
        "--num-workers",
        str(args.num_workers),
        "--epochs",
        "30",
        "--lr-total-epochs",
        "100",
        "--seed",
        str(seed),
        "--run-name",
        run_name(arm, seed),
        "--device",
        args.device,
        "--amp",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def worker(args: argparse.Namespace) -> None:
    root = args.submission_root.resolve()
    logs = root / "logs"
    log_path = logs / "conv1_initialization_ablation_30e.log"
    status_path = logs / "conv1_initialization_ablation_30e_status.json"
    state: dict[str, Any] = {
        "protocol": "flame3_conv1_initialization_ablation_30e",
        "state": "starting",
        "started_unix": time.time(),
        "run_order": [f"seed{seed}:{arm}" for seed, arm in RUN_ORDER],
        "test_split_read": False,
    }
    write_status(status_path, state)
    try:
        configs, preflight_payload = preflight(args)
        state["preflight"] = preflight_payload
        logs.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write("\nFLAME3_CONV1_INITIALIZATION_ABLATION_STARTED\n")
            for seed, arm in RUN_ORDER:
                directory = run_directory(root, configs[arm], arm, seed)
                epochs = metric_epochs(directory / "metrics.jsonl")
                complete = 30 in epochs and (directory / "last.pth").is_file()
                state.update({"state": "running", "current_seed": seed, "current_arm": arm, "current_run": str(directory)})
                write_status(status_path, state)
                if complete:
                    stream.write(f"SKIP_ALREADY_COMPLETE seed{seed} arm{arm}\n")
                    continue
                resume = None
                if directory.exists() and any(directory.iterdir()):
                    candidate = directory / "last.pth"
                    if not candidate.is_file():
                        raise RuntimeError(f"Incomplete run has no last.pth: {directory}")
                    resume = candidate
                completed = subprocess.run(
                    training_command(args, root, arm, seed, resume),
                    cwd=root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    check=False,
                )
                epochs = metric_epochs(directory / "metrics.jsonl")
                if completed.returncode != 0 or 30 not in epochs or not (directory / "last.pth").is_file():
                    raise RuntimeError(f"Training failed or incomplete: seed{seed} arm{arm}")
            summary_command = [
                sys.executable,
                "-m",
                "src.summarize_conv1_ablation_30e",
                "--submission-root",
                str(root),
                "--bundle-root",
                str(args.bundle_root.resolve()),
                "--val-csv",
                str(args.val_csv.resolve()),
                "--annotation-package",
                str(args.annotation_package.resolve()),
                "--device",
                args.device,
                "--amp",
            ]
            summary = subprocess.run(
                summary_command,
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                check=False,
            )
            if summary.returncode != 0:
                raise RuntimeError("Conv1 summarizer failed")
            state.update({"state": "complete", "current_seed": None, "current_arm": None, "completed_unix": time.time()})
            write_status(status_path, state)
            stream.write("FLAME3_CONV1_INITIALIZATION_ABLATION_COMPLETE\n")
    except Exception as error:
        state.update({"state": "failed", "error_type": type(error).__name__, "error": str(error), "failed_unix": time.time()})
        write_status(status_path, state)
        raise


def detach(args: argparse.Namespace) -> None:
    command = [sys.executable, "-m", "src.run_conv1_ablation_30e"]
    for key, value in (
        ("--submission-root", args.submission_root),
        ("--bundle-root", args.bundle_root),
        ("--pretrained", args.pretrained),
        ("--train-csv", args.train_csv),
        ("--val-csv", args.val_csv),
        ("--annotation-package", args.annotation_package),
        ("--device", args.device),
        ("--num-workers", args.num_workers),
    ):
        command.extend((key, str(value)))
    command.append("--worker")
    creationflags = 0
    for name in ("CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS", "CREATE_NO_WINDOW", "CREATE_BREAKAWAY_FROM_JOB"):
        creationflags |= int(getattr(subprocess, name, 0))
    process = subprocess.Popen(
        command,
        cwd=args.submission_root.resolve(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )
    print(f"DETACHED_PID={process.pid}")


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        _, payload = preflight(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    worker(args) if args.worker else detach(args)


if __name__ == "__main__":
    main()

