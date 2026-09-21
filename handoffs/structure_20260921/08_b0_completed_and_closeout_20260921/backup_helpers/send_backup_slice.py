"""Write a fixed archive slice to stdout, optionally rate-limited."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--offset", required=True, type=int)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--rate-mib", default=4.0, type=float)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    size = archive.stat().st_size
    if args.offset < 0 or args.length < 0 or args.offset + args.length > size:
        raise RuntimeError("Requested slice is outside archive")
    rate = args.rate_mib * 1024 * 1024
    started = time.monotonic()
    sent = 0
    output = sys.stdout.buffer
    with archive.open("rb") as stream:
        stream.seek(args.offset)
        remaining = args.length
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise RuntimeError("Unexpected archive EOF")
            output.write(block)
            output.flush()
            sent += len(block)
            remaining -= len(block)
            delay = sent / rate - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
