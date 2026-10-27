"""Compare two run output directories for byte-level determinism.

Compares the Parquet data files byte-for-byte and the ``deterministic`` blocks
of the two manifests. Exits 0 when identical, 1 otherwise (CI gate).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(dir_a: str, dir_b: str) -> int:
    a, b = Path(dir_a), Path(dir_b)
    failures: list[str] = []

    for name in ("train", "val", "test"):
        path_a, path_b = a / "data" / f"{name}.parquet", b / "data" / f"{name}.parquet"
        if not path_a.exists() or not path_b.exists():
            failures.append(f"missing output {name}.parquet")
            continue
        if sha256(path_a) != sha256(path_b):
            failures.append(f"{name}.parquet hashes differ")

    manifest_a = json.loads((a / "artifacts" / "manifest.json").read_text())
    manifest_b = json.loads((b / "artifacts" / "manifest.json").read_text())
    if manifest_a["deterministic"] != manifest_b["deterministic"]:
        keys_a = set(manifest_a["deterministic"])
        keys_b = set(manifest_b["deterministic"])
        differing = [
            key
            for key in keys_a & keys_b
            if manifest_a["deterministic"][key] != manifest_b["deterministic"][key]
        ]
        failures.append(
            f"deterministic manifest blocks differ (keys: {sorted(differing | (keys_a ^ keys_b))})"
        )

    if failures:
        for failure in failures:
            print(f"DETERMINISM FAIL: {failure}")
        return 1
    print("DETERMINISM OK: parquet bytes and deterministic manifest blocks are identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
