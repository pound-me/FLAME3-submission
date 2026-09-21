"""Verify, install and start the hidden SYSTEM task for appended Stage 3."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASK = "FLAME3_Append_Stage3_20260919"
PYTHON = Path(r"C:\Users\Admin\anaconda3\python.exe")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_package():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        if sha(ROOT / row["relative"]) != row["sha256"]:
            raise RuntimeError("Package drift: " + row["relative"])
    return sha(ROOT / "MANIFEST.json")


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError({
            "command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
        })
    return result


def main():
    manifest_sha256 = verify_package()
    action = f'"{PYTHON}" -B -X utf8 -u "{ROOT / "launch_append_stage3.py"}"'
    start = (dt.datetime.now() + dt.timedelta(minutes=2)).strftime("%H:%M")
    run([
        "schtasks", "/Create", "/TN", TASK, "/TR", action,
        "/SC", "ONCE", "/ST", start, "/RU", "SYSTEM", "/RL", "HIGHEST", "/F",
    ])
    run(["schtasks", "/Run", "/TN", TASK])
    record = {
        "task": TASK, "action": action, "manifest_sha256": manifest_sha256,
        "started": True, "no_gui": True, "read_only": True,
        "training_performed": False, "test107_read": False,
    }
    (ROOT / "TASK_INSTALL.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

