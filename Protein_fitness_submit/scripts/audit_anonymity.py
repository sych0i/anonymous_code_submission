"""Fail if the reviewer package contains author-identifying metadata."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "", ".cfg", ".csv", ".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"
}
FORBIDDEN_TEXT = {
    "local Unix home path": re.compile(r"/(?:home|Users|disk\d*)/", re.IGNORECASE),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
}
INFERENCE_CHECKPOINT_KEYS = {
    "format", "backbone", "epoch", "global_step", "pytorch_lightning_version", "state_dict"
}


def identifying_string(value):
    if not isinstance(value, str):
        return None
    for label, pattern in FORBIDDEN_TEXT.items():
        if pattern.search(value):
            return label
    return None


def scan_text_files():
    failures = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", ".venv", "outputs", "validation_outputs", "__pycache__"}
               for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in FORBIDDEN_TEXT.items():
                if pattern.search(line):
                    failures.append(f"{path.relative_to(PACKAGE_ROOT)}:{line_number}: {label}")
    return failures


def walk_metadata(value, location):
    failures = []
    if isinstance(value, torch.Tensor):
        return failures
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(walk_metadata(key, f"{location}.<key>"))
            failures.extend(walk_metadata(item, f"{location}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            failures.extend(walk_metadata(item, f"{location}[{index}]"))
    else:
        label = identifying_string(value)
        if label:
            failures.append(f"{location}: {label}")
    return failures


def is_lfs_pointer(path):
    with open(path, "rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")


def scan_checkpoints():
    failures = []
    for backbone in ("mdlm", "udlm"):
        path = PACKAGE_ROOT / "checkpoints" / backbone / "TrpB" / "best_model.ckpt"
        if is_lfs_pointer(path):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or set(payload) != INFERENCE_CHECKPOINT_KEYS:
            failures.append(f"{path.relative_to(PACKAGE_ROOT)}: unexpected checkpoint metadata keys")
            continue
        if payload.get("format") != "protein-fitness-inference-v1":
            failures.append(f"{path.relative_to(PACKAGE_ROOT)}: not an inference-only checkpoint")
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
        failures.extend(walk_metadata(metadata, str(path.relative_to(PACKAGE_ROOT))))
        del payload

    oracle_dir = PACKAGE_ROOT / "oracle" / "checkpoints" / "TrpB"
    for path in sorted(oracle_dir.glob("*.pth")):
        if is_lfs_pointer(path):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not payload or not all(
                isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in payload.items()):
            failures.append(f"{path.relative_to(PACKAGE_ROOT)}: oracle is not a tensor-only state dict")
    return failures


def audit():
    failures = scan_text_files() + scan_checkpoints()
    if failures:
        raise RuntimeError("anonymity audit failed:\n  " + "\n  ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    audit()
    print("anonymity audit: OK")


if __name__ == "__main__":
    main()
