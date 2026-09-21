"""Create a deterministic source manifest for the stage-3 package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest().upper()


files = []
for path in sorted(ROOT.iterdir(), key=lambda item: item.name.casefold()):
    if path.is_file() and path.name != "MANIFEST.json" and not path.name.endswith(".zip"):
        files.append({"relative": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
authorization = next(item["sha256"] for item in files if item["relative"] == "AUTHORIZATION.md")
payload = {"files": files, "authorization_sha256": authorization}
(ROOT / "MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
