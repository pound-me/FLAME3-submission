"""Preserve an explicit result/source allowlist without opening any dataset."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent
DEST = Path(r"E:\FLAME3_BACKUPS\structure_20260921_v1")
FOLDERS = (
    "stage2_structure_20260918_v1", "stage2_s1r2r3_append_20260919_v1",
    "stage3_structure_20260919_v1", "flame3_append_stage3_20260919_v1",
)
TEXT = {".json", ".jsonl", ".py", ".yaml", ".yml", ".md", ".csv", ".txt", ".ps1", ".cmd"}
WINDOW = {f"epoch{e}.pth" for e in range(26, 31)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def inventory():
    paths = []
    for name in FOLDERS:
        for p in (BUNDLE / name).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if p.name in WINDOW or p.suffix.lower() in TEXT:
                if "test107" in p.as_posix().lower() and p.suffix != ".py":
                    raise RuntimeError("Forbidden backup path: " + str(p))
                paths.append(p)
    cp = [p for p in paths if p.name in WINDOW]
    if len(cp) != 135:
        raise RuntimeError(f"Expected 135 retained candidate checkpoints, found {len(cp)}")
    for seed in (200, 201, 202):
        folder = BUNDLE / "project_support/experiments/flame3_pidnet_s_fusion_manual_smoke_v11" / f"flame3_fusion_manual_smoke_v11_30e_seed{seed}"
        for p in folder.iterdir():
            if p.is_file() and (p.suffix.lower() in TEXT or p.name in ("last.pth", "best_S.pth")):
                paths.append(p)
    paths.append(BUNDLE / "weights/PIDNet_S_ImageNet.pth.tar")
    paths.append(BUNDLE / "project_support/FLAME3_SUBMISSION_SOURCE_20260822/results/efficiency/RTX4090_random_stem_seed200_epoch30.json")
    paths.append(BUNDLE / "project_support/FLAME3_SUBMISSION_SOURCE_20260822/src/evaluate_test107_posthoc.py")
    return sorted(set(paths))


def main():
    if DEST.exists():
        raise FileExistsError("Never overwrite an existing backup: " + str(DEST))
    paths = inventory()
    size = sum(p.stat().st_size for p in paths)
    if shutil.disk_usage(DEST.anchor).free < size * 3:
        raise RuntimeError("Insufficient independent-volume backup space")
    DEST.mkdir(parents=True)
    state = dict(status="COPYING", pid=os.getpid(), files=len(paths), bytes=size,
                 destination=str(DEST), completed=0, test107_read=False, dataset_read=False)
    write(ROOT / "BACKUP_STATUS.json", state)
    rows = []
    for source in paths:
        rel = source.relative_to(BUNDLE)
        target = DEST / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copy2(source, target)
        if sha(target) != before or sha(source) != before:
            raise RuntimeError("Backup/source hash mismatch: " + str(source))
        rows.append(dict(relative=rel.as_posix(), bytes=source.stat().st_size, sha256=before))
        state.update(completed=len(rows), current=rel.as_posix(), updated_unix=time.time())
        write(ROOT / "BACKUP_STATUS.json", state)
    manifest = dict(status="VERIFIED", files=rows, candidate_window_checkpoints=135,
                    source_unchanged=True, test107_read=False, created_unix=time.time())
    write(DEST / "MANIFEST.json", manifest)
    write(ROOT / "BACKUP_MANIFEST.json", manifest)
    archive = DEST / "FLAME3_CHECKPOINT_BACKUP_20260921.zip"
    state.update(status="ARCHIVING", verified_files=len(rows))
    write(ROOT / "BACKUP_STATUS.json", state)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        z.write(DEST / "MANIFEST.json", "MANIFEST.json")
        for row in rows:
            z.write(DEST / "files" / row["relative"], "files/" + row["relative"])
    state.update(status="REMOTE_BACKUP_VERIFIED", archive=str(archive), archive_sha256=sha(archive),
                 archive_bytes=archive.stat().st_size, manifest_sha256=sha(DEST / "MANIFEST.json"),
                 finished_unix=time.time(), off_machine_backup_verified=False)
    write(ROOT / "BACKUP_STATUS.json", state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(ROOT / "BACKUP_FAILURE.json", dict(error=repr(exc), time=time.time()))
        raise
