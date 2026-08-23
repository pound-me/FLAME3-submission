from __future__ import annotations

"""Read-only FLAME3 v2 evaluator for one frozen validation checkpoint.

It computes the unique score S, full-134 Fire/Heat metrics, A001-A047 Smoke
metrics and cross-confusions, plus No-Fire Smoke/Fire/joint false positives.
The script never constructs or reads the test split.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .baseline_runtime import build_dataset, build_model, load_config
from .flame3_manual_smoke_v2_metrics import Flame3ManualSmokeV2Accumulator


ARCHITECTURE_KEYS = (
    "MODEL",
    "MODE",
    "NUM_CLASSES",
    "NUM_OUTPUTS",
    "BACKGROUND_CLASS_INDEX",
    "SMOKE_CLASS_INDEX",
    "FIRE_CLASS_INDEX",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root-dataset", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--annotation-package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Engineering smoke test only; omitted for formal complete validation.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safely_load_training_checkpoint(path: Path) -> dict:
    # Existing project checkpoints also contain NumPy RNG state. Explicitly
    # allow only the narrow NumPy reconstruction types required by that state,
    # while retaining PyTorch's weights_only restricted unpickler.
    safe_types = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
        type(np.dtype(np.float64)),
    ]
    safe_context = getattr(torch.serialization, "safe_globals", None)
    if safe_context is None:
        # PyTorch 2.1 has the restricted weights_only loader but not the
        # safe_globals context manager. New project checkpoints store NumPy
        # RNG state as primitive metadata plus an int64 tensor, so they are
        # directly loadable by the restricted unpickler on that runtime.
        payload = torch.load(path, map_location="cpu", weights_only=True)
    else:
        with safe_context(safe_types):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload is not a dictionary: {type(payload)!r}")
    return payload


def extract_sample_keys(batch_value: object) -> list[str]:
    # Flame3CsvDataset returns [sample_key], so default_collate produces
    # a one-element list containing a tuple of batch keys.
    if isinstance(batch_value, (list, tuple)) and len(batch_value) == 1:
        inner = batch_value[0]
        if isinstance(inner, (list, tuple)):
            return [str(value) for value in inner]
    if isinstance(batch_value, (list, tuple)):
        return [str(value) for value in batch_value]
    raise TypeError(f"Unsupported sample-key batch structure: {type(batch_value)!r}")


def extract_main_logits(outputs: object) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        if not isinstance(outputs[1], torch.Tensor):
            raise TypeError("PIDNet main output is not a tensor")
        return outputs[1]
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def read_val_keys(val_csv: Path) -> list[str]:
    if val_csv.name.lower() == "test.csv" or "test" in val_csv.name.lower():
        raise RuntimeError("Test CSV is forbidden")
    with val_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 134:
        raise RuntimeError(f"Expected 134 validation rows, got {len(rows)}")
    return [row["sample_key"] for row in rows]


def load_manual_targets(
    annotation_package: Path, val_keys: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    freeze_path = (
        annotation_package
        / "final_incremental_audit_20260815"
        / "ANNOTATION_FINAL_FREEZE.json"
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "fully_frozen_for_baseline_and_manual_smoke_v1_training":
        raise RuntimeError("Annotation package is not fully frozen")
    if freeze.get("test_images_or_labels_read") is not False:
        raise RuntimeError("Annotation freeze does not prove test remained sealed")
    manifest_path = annotation_package / "flame3_threeclass_annotation_manifest_150.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets: dict[str, np.ndarray] = {}
    ids: list[str] = []
    for item in manifest["items"]:
        if item["split"] != "val":
            continue
        annotation_id = item["annotation_id"]
        sample_key = item["sample_key"]
        if not annotation_id.startswith("A"):
            raise RuntimeError(f"Malformed annotation ID: {annotation_id}")
        number = int(annotation_id[1:])
        if not 1 <= number <= 47:
            raise RuntimeError(f"Unexpected validation annotation: {annotation_id}")
        if sample_key not in val_keys:
            raise RuntimeError(f"Manual target not in frozen val split: {sample_key}")
        if sample_key in targets:
            raise RuntimeError(f"Duplicate manual validation target: {sample_key}")
        target_path = annotation_package / "completed_masks" / f"{annotation_id}.png"
        target = np.asarray(Image.open(target_path), dtype=np.uint8)
        if target.shape != (512, 640):
            raise RuntimeError(f"Manual target shape mismatch: {annotation_id}")
        targets[sample_key] = target
        ids.append(annotation_id)
    if len(targets) != 47 or sorted(ids) != [f"A{index:03d}" for index in range(1, 48)]:
        raise RuntimeError("A001-A047 manual development targets are incomplete")
    audit = {
        "annotation_manifest": str(manifest_path),
        "annotation_manifest_sha256": sha256_file(manifest_path),
        "annotation_final_freeze": str(freeze_path),
        "annotation_final_freeze_sha256": sha256_file(freeze_path),
        "manual_dev_target_count": len(targets),
        "manual_dev_ids": ids,
    }
    return targets, audit


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    root_dataset = args.root_dataset.resolve()
    val_csv = args.val_csv.resolve()
    annotation_package = args.annotation_package.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    for forbidden in (val_csv, checkpoint_path, output_path):
        if forbidden.name.lower() == "test.csv":
            raise RuntimeError("Test artifact is forbidden")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")


    config = load_config(config_path)
    checkpoint = safely_load_training_checkpoint(checkpoint_path)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        mismatches = {
            key: (config.get(key), checkpoint_config.get(key))
            for key in ARCHITECTURE_KEYS
            if config.get(key) != checkpoint_config.get(key)
        }
        if mismatches:
            raise RuntimeError(f"Config/checkpoint architecture mismatch: {mismatches}")
    config["ROOTDATASET"] = str(root_dataset)
    config["VALIDSET"] = str(val_csv)
    config["DEVICE"] = str(args.device)
    config["NUM_WORKERS"] = int(args.num_workers)
    config["BATCHSIZE"] = int(args.batch_size)

    val_keys = read_val_keys(val_csv)
    if len(val_keys) != len(set(val_keys)):
        raise RuntimeError("Duplicate sample keys in val CSV")
    manual_targets, annotation_audit = load_manual_targets(
        annotation_package, val_keys
    )

    dataset = build_dataset(config, split="val")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=False,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = build_model(config, augment=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    accumulator = Flame3ManualSmokeV2Accumulator()
    completed_batches = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images, labels = batch[0], batch[1]
            sample_keys = extract_sample_keys(batch[3])
            fire_flags = batch[4]
            images = images.to(device=device, dtype=torch.float, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(args.amp and device.type == "cuda"),
            ):
                outputs = model(images)
                logits = extract_main_logits(outputs)
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=bool(config.get("ALIGN_CORNERS", True)),
                    )
            predictions = logits.argmax(dim=1)
            accumulator.update(
                predictions,
                labels,
                sample_keys,
                fire_flags,
                manual_targets,
            )
            completed_batches += 1

    complete = args.max_batches is None
    metrics = accumulator.finalize(require_complete=complete)
    result = {
        "protocol": "flame3_manual_smoke_v2_checkpoint_readonly_evaluation",
        "status": "complete" if complete else "engineering_partial_smoke_test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric_name": checkpoint.get("selection_metric_name"),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "val_csv": str(val_csv),
        "val_csv_sha256": sha256_file(val_csv),
        "val_sample_count": len(dataset),
        "annotation_audit": annotation_audit,
        "runtime": {
            "device": str(device),
            "amp": bool(args.amp and device.type == "cuda"),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "completed_batches": completed_batches,
            "max_batches": args.max_batches,
        },
        "metrics": metrics,
        "training_performed": False,
        "checkpoint_modified": False,
        "test_images_or_labels_read": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
