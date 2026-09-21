"""Capture import-time failures before the stage-3 main exception handler exists."""
from __future__ import annotations

import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAILURE = ROOT / "BOOTSTRAP_FAILURE.log"

if FAILURE.exists():
    FAILURE.unlink()

try:
    sys.argv[0] = str(ROOT / "run_stage3.py")
    runpy.run_path(str(ROOT / "run_stage3.py"), run_name="__main__")
except BaseException:
    FAILURE.write_text(traceback.format_exc(), encoding="utf-8")
    raise
