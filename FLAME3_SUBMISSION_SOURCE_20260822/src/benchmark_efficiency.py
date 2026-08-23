from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline_runtime import build_dataset, build_model, load_config


AUXILIARY_PREFIXES = ("seghead_p.", "seghead_d.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard FLAME3 inference and data-loading efficiency benchmark."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-batches", type=int, default=100)
    return parser.parse_args()


def safe_checkpoint_load(path: Path) -> tuple[dict[str, Any], str]:
    safe_types = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
        type(np.dtype(np.float64)),
    ]
    safe_context = getattr(torch.serialization, "safe_globals", None)
    if safe_context is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        loading_mode = "restricted_weights_only_pytorch_2_1"
    else:
        with safe_context(safe_types):
            payload = torch.load(path, map_location="cpu", weights_only=True)
        loading_mode = "restricted_weights_only_with_safe_globals"
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary")
    return payload, loading_mode


def extract_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise TypeError("Checkpoint state must map string keys to tensors")
    return state


def load_augment_false_model(
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_model(config, augment=False)
    payload, deserialization_mode = safe_checkpoint_load(checkpoint)
    state = extract_state(payload)
    model_state = model.state_dict()
    missing = sorted(set(model_state).difference(state))
    shape_mismatch = sorted(
        key
        for key in set(model_state).intersection(state)
        if model_state[key].shape != state[key].shape
    )
    ignored = sorted(
        key
        for key in set(state).difference(model_state)
        if key.startswith(AUXILIARY_PREFIXES)
    )
    unexpected = sorted(set(state).difference(model_state).difference(ignored))
    if missing or shape_mismatch or unexpected:
        raise RuntimeError(
            "augment=False checkpoint mismatch: "
            f"missing={missing}, shape_mismatch={shape_mismatch}, unexpected={unexpected}"
        )
    model.load_state_dict({key: state[key] for key in model_state}, strict=True)
    model.to(device).eval()
    audit = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "deserialization_mode": deserialization_mode,
        "loaded_tensor_count": len(model_state),
        "ignored_auxiliary_keys": ignored,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": shape_mismatch,
    }
    return model, audit


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def forward_once(model: torch.nn.Module, sample: torch.Tensor, amp: bool) -> None:
    with torch.inference_mode(), torch.autocast(
        device_type=sample.device.type,
        dtype=torch.float16,
        enabled=amp and sample.device.type == "cuda",
    ):
        output = model(sample)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"augment=False must return one tensor, got {type(output)!r}")


def measure_latency(
    model: torch.nn.Module,
    sample: torch.Tensor,
    amp: bool,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, Any]:
    if warmup < 0 or iterations <= 0 or trials <= 0:
        raise ValueError("warmup must be non-negative; iterations and trials must be positive")
    for _ in range(warmup):
        forward_once(model, sample, amp)
    if sample.device.type == "cuda":
        torch.cuda.synchronize(sample.device)
        torch.cuda.reset_peak_memory_stats(sample.device)
    trial_values: list[list[float]] = []
    for _ in range(trials):
        values: list[float] = []
        for _ in range(iterations):
            if sample.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward_once(model, sample, amp)
                end.record()
                end.synchronize()
                values.append(float(start.elapsed_time(end)))
            else:
                start_time = time.perf_counter()
                forward_once(model, sample, amp)
                values.append((time.perf_counter() - start_time) * 1000.0)
        trial_values.append(values)
    flattened = [value for trial in trial_values for value in trial]
    peak_allocated = (
        float(torch.cuda.max_memory_allocated(sample.device) / 1024**2)
        if sample.device.type == "cuda"
        else None
    )
    peak_reserved = (
        float(torch.cuda.max_memory_reserved(sample.device) / 1024**2)
        if sample.device.type == "cuda"
        else None
    )
    return {
        "warmup_iterations": warmup,
        "timed_iterations_per_trial": iterations,
        "trials": trials,
        "sample_count": len(flattened),
        "mean_ms": statistics.fmean(flattened),
        "median_ms": statistics.median(flattened),
        "p95_ms": percentile(flattened, 95),
        "std_ms": statistics.pstdev(flattened),
        "fps_from_mean": 1000.0 / statistics.fmean(flattened),
        "per_trial_mean_ms": [statistics.fmean(values) for values in trial_values],
        "peak_allocated_vram_mb": peak_allocated,
        "peak_reserved_vram_mb": peak_reserved,
    }


def measure_complexity(model: torch.nn.Module, sample: torch.Tensor) -> dict[str, Any]:
    try:
        from thop import profile
    except ImportError as error:
        raise RuntimeError("FLOP measurement requires the pinned thop dependency") from error
    with torch.inference_mode():
        macs, parameters = profile(model, inputs=(sample,), verbose=False)
    return {
        "parameters": int(parameters),
        "macs": int(macs),
        "gmacs": float(macs / 1e9),
        "flops_two_per_mac": int(2 * macs),
        "gflops_two_per_mac": float(2 * macs / 1e9),
        "counting_convention": "THOP MACs; FLOPs additionally reported as 2 x MACs",
    }


def measure_data_loading(
    config: dict[str, Any],
    data_root: Path,
    val_csv: Path,
    device: torch.device,
    num_workers: int,
    batches: int,
) -> dict[str, Any]:
    if "test" in val_csv.name.lower():
        raise RuntimeError("Test split is forbidden in the efficiency benchmark")
    data_config = dict(config)
    data_config["ROOTDATASET"] = str(data_root.resolve())
    data_config["VALIDSET"] = str(val_csv.resolve())
    dataset = build_dataset(data_config, split="val")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    if batches <= 0:
        raise ValueError("data-batches must be positive")
    host_ms: list[float] = []
    transfer_ms: list[float] = []
    iterator = iter(loader)
    for _ in range(min(batches, len(dataset))):
        start = time.perf_counter()
        batch = next(iterator)
        host_ms.append((time.perf_counter() - start) * 1000.0)
        images = batch[0]
        start = time.perf_counter()
        images.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "dataset_rows": len(dataset),
        "measured_batches": len(host_ms),
        "batch_size": 1,
        "num_workers": num_workers,
        "host_loader_mean_ms": statistics.fmean(host_ms),
        "host_loader_p95_ms": percentile(host_ms, 95),
        "host_to_device_mean_ms": statistics.fmean(transfer_ms),
        "host_to_device_p95_ms": percentile(transfer_ms, 95),
    }


def main() -> None:
    args = parse_args()
    if (args.data_root is None) != (args.val_csv is None):
        raise ValueError("--data-root and --val-csv must be supplied together")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = load_config(args.config.resolve())
    config["DEVICE"] = str(device)
    model, loading_audit = load_augment_false_model(
        config, args.checkpoint.resolve(), device
    )
    sample = torch.randn(
        1,
        4,
        args.height,
        args.width,
        device=device,
        dtype=torch.float32,
    )
    complexity = measure_complexity(model, sample)
    latency = measure_latency(
        model,
        sample,
        bool(args.amp),
        args.warmup,
        args.iterations,
        args.trials,
    )
    data_loading = None
    if args.data_root is not None and args.val_csv is not None:
        data_loading = measure_data_loading(
            config,
            args.data_root,
            args.val_csv,
            device,
            args.num_workers,
            args.data_batches,
        )
    result = {
        "protocol": "flame3_submission_efficiency_v1",
        "status": "complete",
        "inference": {
            "augment": False,
            "batch_size": 1,
            "input_height": args.height,
            "input_width": args.width,
            "amp": bool(args.amp and device.type == "cuda"),
            "device": str(device),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "checkpoint_loading": loading_audit,
        "complexity": complexity,
        "latency": latency,
        "data_loading": data_loading,
        "test_split_read": False,
        "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
