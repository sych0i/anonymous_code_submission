#!/usr/bin/env python3
"""
Pairwise Levenshtein (edit) distance for lines in a JSONL file (e.g. text_samples_*.jsonl).

Usage:
  python text_samples_edit_distance.py path/to/text_samples_p5_r3_allstep.jsonl
  python text_samples_edit_distance.py path/to/file.jsonl --unique-threshold 0.05
  python text_samples_edit_distance.py path/to/file.jsonl --normalize max_len --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

__all__ = (
    "levenshtein",
    "levenshtein_bitparallel",
    "unique_count_clustering",
    "pairwise_levenshtein_raw",
    "pairwise_levenshtein_bounded",
    "unique_count_for_texts",
)


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein distance; O(len(a)*len(b)) time, O(min) space."""
    m, n = len(a), len(b)
    if m < n:
        a, b = b, a
        m, n = n, m
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ca = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def levenshtein_bitparallel(a: str, b: str) -> int:
    """Exact Levenshtein distance using Myers' bit-vector algorithm."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a

    char_masks: dict[str, int] = {}
    for index, char in enumerate(a):
        char_masks[char] = char_masks.get(char, 0) | (1 << index)

    positive = ~0
    negative = 0
    score = len(a)
    last = 1 << (len(a) - 1)
    for char in b:
        equal = char_masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & last:
            score += 1
        elif negative_horizontal & last:
            score -= 1
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(vertical | positive_horizontal)
        negative = positive_horizontal & vertical
    return score


def _levenshtein_at_most(a: str, b: str, limit: int) -> int:
    """Return Levenshtein distance, or limit + 1 once it is known to exceed limit."""
    if limit < 0:
        return limit + 1
    m, n = len(a), len(b)
    if m < n:
        a, b = b, a
        m, n = n, m
    if n == 0:
        return m if m <= limit else limit + 1
    if m - n > limit:
        return limit + 1

    inf = limit + 1
    prev = [inf] * (n + 1)
    for j in range(min(n, limit) + 1):
        prev[j] = j

    for i in range(1, m + 1):
        lo = max(1, i - limit)
        hi = min(n, i + limit)
        if lo > hi:
            return inf

        cur = [inf] * (n + 1)
        if i <= limit:
            cur[0] = i
        ca = a[i - 1]
        row_min = cur[0]
        for j in range(lo, hi + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > limit:
            return inf
        prev = cur

    return prev[n] if prev[n] <= limit else inf


def load_texts(path: Path, field: str, strip_eot: bool) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {e}") from e
            if field not in obj:
                raise SystemExit(f"{path}:{line_no}: missing key {field!r}")
            t = obj[field]
            if not isinstance(t, str):
                raise SystemExit(f"{path}:{line_no}: {field!r} must be str, got {type(t).__name__}")
            if strip_eot:
                t = t.replace("<|endoftext|>", "")
            texts.append(t)
    if len(texts) == 0:
        raise SystemExit("no non-empty JSON lines found")
    return texts


class _DSU:
    """Disjoint-set for transitive duplicate merging."""

    __slots__ = ("p",)

    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _norm_dist(
    d: int, texts: list[str], i: int, j: int, mode: str
) -> float:
    if mode == "max_len":
        denom = max(len(texts[i]), len(texts[j]))
    else:
        denom = (len(texts[i]) + len(texts[j])) / 2.0
    return float(d) / denom if denom else 0.0


def unique_count_clustering(
    texts: list[str], raw_ds: list[int], threshold: float, normalize: str
) -> tuple[int, list[list[int]]]:
    """
    Merge lines i,j when normalized Levenshtein <= threshold (transitive).
    Returns (number of clusters, list of cluster index lists).
    """
    n = len(texts)
    if n == 0:
        return 0, []
    if n == 1:
        return 1, [[0]]

    dsu = _DSU(n)
    for (i, j), d in zip(combinations(range(n), 2), raw_ds):
        if _norm_dist(d, texts, i, j, normalize) <= threshold:
            dsu.union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        r = dsu.find(i)
        buckets.setdefault(r, []).append(i)
    clusters = sorted(buckets.values(), key=lambda c: min(c))
    return len(clusters), clusters


def pairwise_levenshtein_raw(texts: list[str]) -> list[int]:
    """Upper-triangle pairwise Levenshtein distances, same order as combinations(range(n), 2)."""
    n = len(texts)
    if n < 2:
        return []
    return [levenshtein(texts[i], texts[j]) for i, j in combinations(range(n), 2)]


def pairwise_levenshtein_bounded(
    texts: list[str], max_threshold: float, normalize: str = "max_len"
) -> list[int]:
    """
    Pairwise distances needed for clustering up to ``max_threshold``.

    Values above each pair's threshold are represented by limit + 1; this is
    sufficient for every clustering threshold <= ``max_threshold``.
    """
    if max_threshold < 0:
        raise ValueError("max_threshold must be non-negative")
    distances = []
    for i, j in combinations(range(len(texts)), 2):
        if normalize == "max_len":
            denominator = max(len(texts[i]), len(texts[j]))
        else:
            denominator = (len(texts[i]) + len(texts[j])) / 2.0
        limit = math.floor(max_threshold * denominator + 1e-12)
        distance = levenshtein_bitparallel(texts[i], texts[j])
        distances.append(distance if distance <= limit else limit + 1)
    return distances


def unique_count_for_texts(
    texts: list[str], threshold: float, normalize: str = "max_len"
) -> int:
    """
    Number of clusters when merging strings with normalized edit distance <= threshold
    (transitive closure). Empty list -> 0; one string -> 1.
    """
    n = len(texts)
    if n == 0:
        return 0
    if n == 1:
        return 1

    dsu = _DSU(n)
    for i, j in combinations(range(n), 2):
        if normalize == "max_len":
            denom = max(len(texts[i]), len(texts[j]))
        else:
            denom = (len(texts[i]) + len(texts[j])) / 2.0
        limit = math.floor(threshold * denom + 1e-12)
        if _levenshtein_at_most(texts[i], texts[j], limit) <= limit:
            dsu.union(i, j)

    return len({dsu.find(i) for i in range(n)})


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise edit distance for JSONL text field.")
    p.add_argument("jsonl", type=Path, help="Path to .jsonl (one JSON object per line)")
    p.add_argument(
        "--field",
        default="text",
        help='JSON key to compare (default: "text")',
    )
    p.add_argument(
        "--strip-eot",
        action="store_true",
        help='Remove "<|endoftext|>" substrings before distance (optional)',
    )
    p.add_argument(
        "--normalize",
        choices=("none", "max_len", "mean_len"),
        default="none",
        help="Divide raw distance by max or mean of the two string lengths (default: none)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write pairwise distances to this CSV file",
    )
    p.add_argument(
        "--unique-threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "If set, count unique texts by merging pairs whose normalized Levenshtein "
            "distance is <= T (transitive closure). Typical range: 0.02–0.1"
        ),
    )
    p.add_argument(
        "--unique-normalize",
        choices=("max_len", "mean_len"),
        default="max_len",
        help="Normalization for --unique-threshold (default: max_len)",
    )
    args = p.parse_args()

    texts = load_texts(args.jsonl, args.field, args.strip_eot)
    n = len(texts)

    raw_ds: list[int] = []
    if n >= 2:
        for i, j in combinations(range(n), 2):
            raw_ds.append(levenshtein(texts[i], texts[j]))

    print(f"file: {args.jsonl}")
    print(f"lines (texts): {n}")
    if n >= 2:
        mean_raw = sum(raw_ds) / len(raw_ds)
        print(f"pairwise pairs: {len(raw_ds)}")
        print(f"mean Levenshtein (raw): {mean_raw:.4f}")
        if args.normalize != "none":
            nds: list[float] = []
            for (i, j), d in zip(combinations(range(n), 2), raw_ds):
                if args.normalize == "max_len":
                    denom = max(len(texts[i]), len(texts[j]))
                else:
                    denom = (len(texts[i]) + len(texts[j])) / 2.0
                nds.append(float(d) / denom if denom else 0.0)
            print(f"mean normalized ({args.normalize}): {sum(nds) / len(nds):.6f}")

    if args.unique_threshold is not None:
        if args.unique_threshold < 0:
            raise SystemExit("--unique-threshold must be >= 0")
        if n >= 2:
            uq, clusters = unique_count_clustering(
                texts, raw_ds, args.unique_threshold, args.unique_normalize
            )
        else:
            uq, clusters = 1, [[0]]
        print("\n--- Uniqueness (normalized edit distance clustering) ---")
        print(f"threshold: <= {args.unique_threshold}  ({args.unique_normalize})")
        print(f"unique count: {uq}  (out of {n} lines)")
        for cidx, idxs in enumerate(clusters):
            print(f"  cluster {cidx}: indices {idxs}  (size {len(idxs)})")

    # Compact distance matrix (symmetric)
    if n >= 2:
        print("\nFull distance matrix (raw Levenshtein):")
        header = "idx " + " ".join(f"{j:>8}" for j in range(n))
        print(header)
        for i in range(n):
            row = [levenshtein(texts[i], texts[j]) if j != i else 0 for j in range(n)]
            print(f"{i:>3} " + " ".join(f"{v:>8}" for v in row))
    else:
        print("\nFull distance matrix: single line (skipped)")

    if args.csv:
        if n < 2:
            raise SystemExit("--csv requires at least 2 text lines")
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["i", "j", "distance_raw", f"distance_{args.normalize}"])
            for (i, j), dr in zip(combinations(range(n), 2), raw_ds):
                if args.normalize == "none":
                    dn = ""
                elif args.normalize == "max_len":
                    denom = max(len(texts[i]), len(texts[j]))
                    dn = f"{float(dr) / denom:.8f}" if denom else "0"
                else:
                    denom = (len(texts[i]) + len(texts[j])) / 2.0
                    dn = f"{float(dr) / denom:.8f}" if denom else "0"
                w.writerow([i, j, dr, dn])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
