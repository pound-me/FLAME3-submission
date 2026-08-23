from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PATH_FIELDS = (
    "corrected_rgb_path",
    "raw_thermal_path",
    "thermal_tiff_path",
    "temperature_mask_path",
    "boundary_mask_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the portable per-sample FLAME3 split and file-hash manifest."
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = parse_args()
    root = args.bundle_root.resolve()
    inputs = [("train", args.train_csv.resolve()), ("val", args.val_csv.resolve())]
    if args.test_csv is not None:
        inputs.append(("test107", args.test_csv.resolve()))
    expected_counts = {"train": 493, "val": 134, "test107": 107}
    manifest_rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for split, csv_path in inputs:
        rows = read_rows(csv_path)
        if len(rows) != expected_counts[split]:
            raise RuntimeError(
                f"{split} row count mismatch: {len(rows)} != {expected_counts[split]}"
            )
        for row in rows:
            sample_key = str(row["sample_key"])
            if sample_key in seen_keys:
                raise RuntimeError(f"Sample appears in multiple splits: {sample_key}")
            seen_keys.add(sample_key)
            dense = int(row.get("dense_supervision", "0") or 0)
            item: dict[str, Any] = {
                "split": split,
                "sample_key": sample_key,
                "sample_class": row.get("sample_class"),
                "sample_id": row.get("sample_id"),
                "dense_supervision": dense,
                "supervision_source": "manual_dense" if dense else "machine_partial",
                "thermal_tiff_role": "diagnostic_only_not_model_input",
            }
            for field in PATH_FIELDS:
                value = str(row.get(field, "") or "")
                if not value:
                    item[f"{field}_path"] = ""
                    item[f"{field}_sha256"] = ""
                    continue
                path = resolve(root, value)
                item[f"{field}_path"] = value.replace("\\", "/")
                if not path.is_file():
                    missing.append(
                        {"split": split, "sample_key": sample_key, "field": field, "path": str(path)}
                    )
                    item[f"{field}_sha256"] = ""
                else:
                    item[f"{field}_sha256"] = sha256_file(path)
            manifest_rows.append(item)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} referenced files are missing; first entries: {missing[:10]}"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(manifest_rows[0])
    csv_output = output / "FLAME3_COMPLETE_SPLIT_AND_FILE_HASH_MANIFEST.csv"
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "protocol": "flame3_complete_split_and_file_hash_manifest_v1",
        "status": "complete",
        "bundle_root_recorded_for_audit_only": str(root),
        "split_counts": {
            split: sum(row["split"] == split for row in manifest_rows)
            for split, _ in inputs
        },
        "dense_supervision_counts": {
            split: sum(row["split"] == split and row["dense_supervision"] == 1 for row in manifest_rows)
            for split, _ in inputs
        },
        "unique_sample_count": len(seen_keys),
        "manifest_csv": csv_output.name,
        "manifest_csv_sha256": sha256_file(csv_output),
        "source_csv_sha256": {
            split: sha256_file(csv_path) for split, csv_path in inputs
        },
        "test_included": args.test_csv is not None,
        "training_or_inference_performed": False,
    }
    (output / "FLAME3_COMPLETE_SPLIT_AND_FILE_HASH_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
