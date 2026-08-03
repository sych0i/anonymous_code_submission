#!/usr/bin/env python3
"""Verify submission artifacts without requiring a GPU or the MNIST dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_portable_warmup import prepare  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    expected = json.loads((ROOT / "expected_results.json").read_text(encoding="utf-8"))
    hashes = expected["artifact_hashes"]
    files = {
        "model_checkpoint_sha256": ROOT / "image_exp/models/model060000.pt",
        "classifier_checkpoint_sha256": ROOT / "image_exp/models/resnet.pth.tar",
        "model_config_sha256": ROOT / "image_exp/image_confs/mnist_model_and_diffusion_conf.yml",
        "task_config_sha256": ROOT / "image_exp/image_confs/task_class_cond_gen_conf.yml",
        "warmup_value_means_sha256": ROOT / "artifacts/shared_warmup/value_means.json",
    }
    failures = []
    for key, path in files.items():
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual = sha256(path)
        if actual != hashes[key]:
            failures.append(f"{path}: {actual} != {hashes[key]}")
    if failures:
        raise SystemExit("submission hash verification failed:\n" + "\n".join(failures))

    portable = prepare()
    implementation = json.loads(portable.read_text(encoding="utf-8"))["warmup_config"][
        "provenance"
    ]["implementation_sha256"]
    if implementation != hashes["implementation_sha256"]:
        raise SystemExit(
            "implementation fingerprint mismatch: "
            f"{implementation} != {hashes['implementation_sha256']}"
        )
    print(f"Verified {len(files)} fixed artifact hashes.")
    print(f"Verified implementation fingerprint: {implementation}")
    print(f"Verified portable warmup: {portable}")
    print("Setup verification passed (GPU sampling was not run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
