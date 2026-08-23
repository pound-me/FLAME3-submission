from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from third_party.RoboFireFuseNet.models.pidnet import PIDNet
from third_party.RoboFireFuseNet.utils.total_loss import TotalLoss
from .flame3_dataset import Flame3CsvDataset

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dataset(config: dict, split: str):
    split_to_key = {
        "train": "TRAINSET",
        "val": "VALIDSET",
        "test": "TESTSET",
    }
    if split not in split_to_key:
        raise ValueError(f"Unsupported split: {split}")
    training = split == "train"
    if str(config.get("DATASET_TYPE", "wildfire_txt")).lower() == "flame3_csv":
        if config.get("TRAINING_OBJECTIVE") != "partial_label":
            raise ValueError(
                "FLAME3 CSV training requires TRAINING_OBJECTIVE=partial_label"
            )
        return Flame3CsvDataset(
            root=config["ROOTDATASET"],
            csv_path=config[split_to_key[split]],
            mode=config["MODE"],
            multi_scale=config["MULTISCALE"] if training else False,
            flip=config["FLIP"] if training else False,
            brightness=config["BRIGHTNESS"] if training else False,
            ignore_label=config["IGNORE_LABEL"],
            scale_factor=config["SCALE_FACTOR"],
            crop_size=config["CROP_SIZE"],
            base_size=config["BASE_SIZE"],
            bd_dilate_size=int(config.get("BD_DILATE_SIZE", 4)),
            comp_mask=config["COMP_MASK"] if training else False,
            single_source=config["SINGLE_SOURCE"] if training else False,
            scale_min=float(config.get("SCALE_MIN", 0.8)),
            scale_max=float(config.get("SCALE_MAX", 1.5)),
            protect_fire_core_crop=(
                training
                and bool(config.get("FLAME3_PROTECT_FIRE_CORE_CROP", False))
            ),
            fire_core_crop_min_pixels=int(
                config.get("FLAME3_FIRE_CORE_CROP_MIN_PIXELS", 1)
            ),
            fire_core_crop_attempts=int(
                config.get("FLAME3_FIRE_CORE_CROP_ATTEMPTS", 32)
            ),
        )
    raise ValueError(
        "Submission runtime supports DATASET_TYPE=flame3_csv only; "
        f"got {config.get('DATASET_TYPE')!r}."
    )


def build_model(config: dict, augment: bool = True) -> PIDNet:
    model_name = str(config["MODEL"])
    if model_name != "pidnet_s":
        raise ValueError(
            "Submission source contains the frozen PIDNet-S method only; "
            f"got MODEL={model_name!r}."
        )
    input_channels = {"rgb": 3, "ir": 1, "fusion": 4}
    mode = str(config["MODE"]).lower()
    if mode not in input_channels:
        raise ValueError(f"Unsupported input mode: {config['MODE']}")
    return PIDNet(
        m=2,
        n=3,
        num_classes=int(config["NUM_CLASSES"]),
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=augment,
        channels=input_channels[mode],
    )


def _checkpoint_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Pretrained checkpoint must be a dictionary")
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError("Pretrained state must map string keys to tensors")
    return state


def _inflate_rgb_stem_with_ir_mean(
    source: torch.Tensor,
    destination: torch.Tensor,
) -> torch.Tensor:
    if source.ndim != 4 or destination.ndim != 4:
        raise ValueError("Stem tensors must be four-dimensional convolution weights")
    if source.shape[1] != 3 or destination.shape[1] != 4:
        raise ValueError(
            "rgb_copy_ir_mean requires ImageNet 3-channel source and "
            f"4-channel Fusion destination; got {tuple(source.shape)} -> "
            f"{tuple(destination.shape)}"
        )
    if source.shape[0] != destination.shape[0] or source.shape[2:] != destination.shape[2:]:
        raise ValueError(
            "Stem output/kernel geometry mismatch: "
            f"{tuple(source.shape)} -> {tuple(destination.shape)}"
        )
    inflated = destination.new_empty(destination.shape)
    inflated[:, :3].copy_(source.to(dtype=destination.dtype))
    inflated[:, 3:4].copy_(
        source.mean(dim=1, keepdim=True).to(dtype=destination.dtype)
    )
    return inflated


def load_pretrained_if_available(model: PIDNet, config: dict) -> int:
    configured_path = config.get("PRETRAINED")
    stem_key = "conv1.0.weight"
    stem_strategy = str(config.get("PRETRAIN_STEM_STRATEGY", "random"))
    allowed_strategies = {"random", "rgb_copy_ir_mean"}
    if stem_strategy not in allowed_strategies:
        raise ValueError(
            f"PRETRAIN_STEM_STRATEGY must be one of {sorted(allowed_strategies)}, "
            f"got {stem_strategy!r}"
        )
    if not configured_path:
        audit = {
            "enabled": False,
            "stem_strategy": stem_strategy,
            "matched_tensor_count": 0,
            "loaded_tensor_keys": [],
            "explicit_skip_keys": [],
            "skipped_source_keys": [],
        }
        config["PRETRAIN_LOAD_AUDIT"] = audit
        print("Pretrained initialization: disabled")
        return 0

    checkpoint_path = Path(configured_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_state = _checkpoint_state(checkpoint)
    model_state = model.state_dict()

    configured_skip_keys = config.get("PRETRAIN_SKIP_KEYS", [])
    if isinstance(configured_skip_keys, str):
        configured_skip_keys = [configured_skip_keys]
    skip_keys = {str(key) for key in configured_skip_keys}
    unknown_source_keys = sorted(skip_keys.difference(source_state))
    unknown_model_keys = sorted(skip_keys.difference(model_state))
    if unknown_source_keys or unknown_model_keys:
        raise KeyError(
            "Invalid PRETRAIN_SKIP_KEYS: "
            f"missing_from_checkpoint={unknown_source_keys}, "
            f"missing_from_model={unknown_model_keys}"
        )
    if stem_strategy == "random" and stem_key not in skip_keys:
        raise ValueError(
            "PRETRAIN_STEM_STRATEGY=random requires conv1.0.weight in "
            "PRETRAIN_SKIP_KEYS so the complete four-channel stem remains Kaiming-random"
        )
    if stem_strategy == "rgb_copy_ir_mean" and stem_key in skip_keys:
        raise ValueError(
            "PRETRAIN_STEM_STRATEGY=rgb_copy_ir_mean requires conv1.0.weight "
            "to be removed from PRETRAIN_SKIP_KEYS"
        )

    matched_state = {
        key: value
        for key, value in source_state.items()
        if key in model_state
        and model_state[key].shape == value.shape
        and key not in skip_keys
    }
    stem_audit: dict[str, object]
    if stem_strategy == "rgb_copy_ir_mean":
        if stem_key not in source_state or stem_key not in model_state:
            raise KeyError(f"Missing required stem tensor: {stem_key}")
        matched_state[stem_key] = _inflate_rgb_stem_with_ir_mean(
            source_state[stem_key], model_state[stem_key]
        )
        stem_audit = {
            "key": stem_key,
            "source_shape": list(source_state[stem_key].shape),
            "destination_shape": list(model_state[stem_key].shape),
            "rgb_channels": "copied_exactly_from_imagenet",
            "ir_channel": "mean_across_imagenet_rgb_input_channels",
        }
    else:
        stem_audit = {
            "key": stem_key,
            "destination_shape": list(model_state[stem_key].shape),
            "initialization": "pidnet_kaiming_random",
        }

    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=True)
    loaded_keys = sorted(matched_state)
    skipped_source_keys = sorted(set(source_state).difference(loaded_keys))
    shape_mismatch_keys = sorted(
        key
        for key in set(source_state).intersection(model_state)
        if source_state[key].shape != model_state[key].shape and key not in loaded_keys
    )
    audit = {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "source_tensor_count": len(source_state),
        "model_tensor_count": len(model_state),
        "matched_tensor_count": len(matched_state),
        "loaded_tensor_keys": loaded_keys,
        "explicit_skip_keys": sorted(skip_keys),
        "shape_mismatch_keys": shape_mismatch_keys,
        "skipped_source_keys": skipped_source_keys,
        "stem_strategy": stem_strategy,
        "stem": stem_audit,
    }
    config["PRETRAIN_LOAD_AUDIT"] = audit
    print(
        f"Pretrained initialization: {checkpoint_path} "
        f"({len(matched_state)} tensors matched; stem={stem_strategy})"
    )
    if skipped_source_keys:
        print(f"Skipped pretrained tensors: {skipped_source_keys}")
    return len(matched_state)

__all__ = [
    "PROJECT_ROOT",
    "TotalLoss",
    "build_dataset",
    "build_model",
    "load_config",
    "load_pretrained_if_available",
    "seed_everything",
]





