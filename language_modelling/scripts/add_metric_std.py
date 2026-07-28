#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.group_metric_stats import (  # noqa: E402
    group_distinctness,
    group_means,
    group_perplexities,
    group_unique_ratio_sweep,
    particle_group_slices,
    replace_metric_std_section,
)
from model_revisions import pretrained_kwargs  # noqa: E402


THRESHOLDS = (0.05,)
CLASSIFIERS = {
    "cola": "textattack/roberta-base-CoLA",
    "toxic": "SkolkovoInstitute/roberta_toxicity_classifier",
    "toxic_ext": "textdetox/xlmr-large-toxicity-classifier",
}


@dataclass
class Artifact:
    generation_path: Path
    result_path: Path
    ppl_path: Path
    outputs: list[str]
    prompted_texts: list[str]
    group_slices: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add per-(prompt, run) metric mean/std to completed evaluations."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_artifact(path: Path, num_particles: int) -> Artifact:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    outputs = []
    prompted_texts = []
    row_lengths = []
    for row in rows:
        strings = list(row["string"])
        row_lengths.append(len(strings))
        outputs.extend(strings)
        prompted_texts.extend(
            f'{row["context_string"]}{output}' for output in strings
        )

    strategy = path.stem.removeprefix("abc_ssdlm_gen_merged_")
    result_path = path.parent / f"eval_results_{strategy}.txt"
    ppl_path = path.parent / f"eval_results_{strategy}.txt.ppl-gpt2-xl"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    if not ppl_path.exists():
        raise FileNotFoundError(ppl_path)

    return Artifact(
        generation_path=path,
        result_path=result_path,
        ppl_path=ppl_path,
        outputs=outputs,
        prompted_texts=prompted_texts,
        group_slices=particle_group_slices(row_lengths, num_particles),
    )


def file_signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: Path, cache: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))
    temp_path.replace(path)


def predict_labels(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    desc: str,
) -> list[int]:
    labels = []
    starts = range(0, len(texts), batch_size)
    for start in tqdm(starts, desc=desc, leave=False):
        batch = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            predicted = model(**batch).logits.argmax(dim=-1).tolist()
        labels.extend(int(label == 1) for label in predicted)
    return labels


def ensure_classifier_labels(
    artifacts: list[Artifact],
    root: Path,
    cache_path: Path,
    cache: dict,
    classifier_name: str,
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> None:
    pending = []
    for artifact in artifacts:
        key = str(artifact.generation_path.relative_to(root))
        entry = cache.get(key, {})
        labels = entry.get(classifier_name)
        if (
            entry.get("signature") != file_signature(artifact.generation_path)
            or not isinstance(labels, list)
            or len(labels) != len(artifact.prompted_texts)
        ):
            pending.append(artifact)
    if not pending:
        print(f"[cache] {classifier_name}: all {len(artifacts)} files", flush=True)
        return

    print(
        f"[load] {classifier_name}: {model_name} ({len(pending)} files)",
        flush=True,
    )
    revision_kwargs = pretrained_kwargs(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **revision_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **revision_kwargs
    ).to(device)
    model.eval()

    for index, artifact in enumerate(pending, start=1):
        key = str(artifact.generation_path.relative_to(root))
        labels = predict_labels(
            artifact.prompted_texts,
            tokenizer,
            model,
            device,
            batch_size,
            f"{classifier_name} {index}/{len(pending)}",
        )
        entry = cache.setdefault(key, {})
        entry["signature"] = file_signature(artifact.generation_path)
        entry[classifier_name] = labels
        save_cache(cache_path, cache)

    del model
    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def read_ppl_details(path: Path) -> tuple[list[float], list[float], list[int]]:
    ppls = []
    nlls = []
    token_counts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ppl, nll, token_count = (part.strip() for part in line.split(","))
            ppls.append(float(ppl))
            nlls.append(float(nll))
            token_counts.append(int(token_count))
    return ppls, nlls, token_counts


def metric_groups_for(
    artifact: Artifact,
    labels: dict[str, list[int]],
    num_particles: int,
) -> dict[str, list[float]]:
    ppls, nlls, token_counts = read_ppl_details(artifact.ppl_path)
    if len(ppls) != len(artifact.outputs):
        raise ValueError(
            f"{artifact.ppl_path}: {len(ppls)} PPL rows for "
            f"{len(artifact.outputs)} generations"
        )

    group_ppls, group_total_ppls = group_perplexities(
        ppls, nlls, token_counts, artifact.group_slices
    )
    dist1, dist2, dist3 = group_distinctness(
        artifact.outputs, artifact.group_slices
    )
    metrics = {
        "gpt2-xl perplexity": group_ppls,
        "gpt2-xl total perplexity": group_total_ppls,
        "cola acceptability accuracy": group_means(
            labels["cola"], artifact.group_slices
        ),
        "dist-1": dist1,
        "dist-2": dist2,
        "dist-3": dist3,
        "toxic acc": group_means(labels["toxic"], artifact.group_slices),
        "toxic_ext acc": group_means(
            labels["toxic_ext"], artifact.group_slices
        ),
    }
    unique_sweeps = {
        "all": group_unique_ratio_sweep(
            artifact.prompted_texts,
            artifact.group_slices,
            THRESHOLDS,
            num_particles,
        ),
        "toxic": group_unique_ratio_sweep(
            artifact.prompted_texts,
            artifact.group_slices,
            THRESHOLDS,
            num_particles,
            labels["toxic"],
        ),
        "toxic_ext": group_unique_ratio_sweep(
            artifact.prompted_texts,
            artifact.group_slices,
            THRESHOLDS,
            num_particles,
            labels["toxic_ext"],
        ),
    }
    for threshold in THRESHOLDS:
        prefix = f"unique_edit#{threshold:g}"
        count_prefix = f"unique_edit_count#{threshold:g}"
        for subset, sweep in unique_sweeps.items():
            metrics[f"{prefix} {subset}"] = sweep[threshold]
        for subset, sweep in unique_sweeps.items():
            ratios = sweep[threshold]
            metrics[f"{count_prefix} {subset}"] = [
                ratio * num_particles for ratio in ratios
            ]
    return metrics


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    generation_paths = sorted(root.rglob("abc_ssdlm_gen_merged_*.jsonl"))
    if not generation_paths:
        raise FileNotFoundError(f"No merged generation files under {root}")

    artifacts = [
        load_artifact(path, args.num_particles) for path in generation_paths
    ]
    print(f"[found] {len(artifacts)} completed strategies", flush=True)

    cache_path = root / ".metric_std_label_cache.json"
    cache = load_cache(cache_path)
    device = torch.device(args.device)
    for classifier_name, model_name in CLASSIFIERS.items():
        ensure_classifier_labels(
            artifacts,
            root,
            cache_path,
            cache,
            classifier_name,
            model_name,
            device,
            args.batch_size,
        )

    for index, artifact in enumerate(artifacts, start=1):
        key = str(artifact.generation_path.relative_to(root))
        labels = {
            classifier_name: cache[key][classifier_name]
            for classifier_name in CLASSIFIERS
        }
        metric_groups = metric_groups_for(
            artifact, labels, args.num_particles
        )
        replace_metric_std_section(
            artifact.result_path, metric_groups, args.num_particles
        )
        print(
            f"[write {index}/{len(artifacts)}] {artifact.result_path}",
            flush=True,
        )

    print(f"[done] updated {len(artifacts)} result files", flush=True)


if __name__ == "__main__":
    main()
