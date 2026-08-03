"""Verify checksums, anonymity, and the committed reference-table schema."""

import hashlib
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.aggregate_tables import load_reference
from scripts.audit_anonymity import audit


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums():
    manifest = PACKAGE_ROOT / "CHECKSUMS.sha256"
    failures = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"line {line_number}: malformed checksum record")
            continue
        path = PACKAGE_ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256_file(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        raise RuntimeError("checksum verification failed:\n  " + "\n  ".join(failures))


def main():
    verify_checksums()
    audit()
    load_reference(PACKAGE_ROOT / "results" / "reference_tables.json")
    print("package verification: OK")


if __name__ == "__main__":
    main()
