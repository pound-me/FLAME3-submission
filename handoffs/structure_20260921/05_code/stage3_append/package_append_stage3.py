"""Build the immutable appended Stage 3 manifest and transfer archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME_DIRS = {"results", "__pycache__"}
RUNTIME_FILES = {
    "MANIFEST.json", "PREFLIGHT.json", "PROBE_STATUS.json",
    "APPEND_STAGE3_STATUS.json", "APPEND_STAGE3_LAUNCH_STATUS.json",
    "APPEND_STAGE3_FAILURE.json", "APPEND_STAGE3_SUMMARY.json",
    "APPEND_STAGE3_RESULTS.csv", "ATTRIBUTION_2X2.json", "ATTRIBUTION_2X2.csv",
    "TASK_INSTALL.json", "append_stage3_worker.log",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def files():
    for path in sorted(ROOT.rglob("*"), key=lambda item: item.as_posix().casefold()):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in RUNTIME_DIRS for part in relative.parts):
            continue
        if path.name in RUNTIME_FILES or path.suffix.lower() in {".pyc", ".zip"}:
            continue
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        {"relative": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in files()
    ]
    authorization = next(row["sha256"] for row in rows if row["relative"] == "AUTHORIZATION.md")
    manifest = {
        "protocol": "flame3_append_stage3_and_2x2_20260919_v1",
        "authorization_sha256": authorization,
        "training_performed": False, "test107_read": False,
        "files": rows,
    }
    manifest_path = ROOT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    if args.zip.exists():
        args.zip.unlink()
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(manifest_path, "MANIFEST.json")
        for row in rows:
            archive.write(ROOT / row["relative"], row["relative"])
    print(json.dumps({
        "file_count": len(rows), "manifest_sha256": sha(manifest_path),
        "zip_sha256": sha(args.zip), "zip": str(args.zip.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

