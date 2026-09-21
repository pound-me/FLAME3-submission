"""Acquire the shared GPU lock and run the appended read-only Stage 3 evaluation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = ROOT.parent / "flame3_structure_screen_20260917_gpu.lock"
STATUS = ROOT / "APPEND_STAGE3_LAUNCH_STATUS.json"


def write(value):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main():
    descriptor = None
    try:
        descriptor = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"append_stage3 pid={os.getpid()} started={time.time()}\n".encode())
        os.close(descriptor)
        descriptor = None
        write({
            "status": "RUNNING", "pid": os.getpid(),
            "acquired_gpu_lock_unix": time.time(), "training_performed": False,
            "test107_read": False,
        })
        with (ROOT / "append_stage3_worker.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-B", "-X", "utf8", "-u", "run_append_stage3.py"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        write({
            "status": "COMPLETE" if result.returncode == 0 else "FAILED",
            "pid": os.getpid(), "exit_code": result.returncode,
            "finished_unix": time.time(), "training_performed": False,
            "test107_read": False,
        })
        raise SystemExit(result.returncode)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if LOCK.exists():
            try:
                if "append_stage3" in LOCK.read_text(encoding="utf-8"):
                    LOCK.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

