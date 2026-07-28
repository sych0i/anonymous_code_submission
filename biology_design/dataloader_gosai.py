"""DNA tokenization helpers used by the evaluation code."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DNA_ALPHABET = {"A": 0, "C": 1, "G": 2, "T": 3}
INDEX_TO_DNA = np.asarray(["A", "C", "G", "T"])


def dna_detokenize(seq: Sequence[int]) -> str:
    return "".join(INDEX_TO_DNA[np.asarray(seq, dtype=int)])


def batch_dna_detokenize(batch_seq) -> list[str]:
    decoded = INDEX_TO_DNA[np.asarray(batch_seq, dtype=int)]
    return ["".join(seq) for seq in decoded]


def dna_tokenize(seq: str) -> list[int]:
    return [DNA_ALPHABET[base] for base in seq]


def batch_dna_tokenize(batch_seq: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[DNA_ALPHABET[base] for base in seq] for seq in batch_seq],
        dtype=np.int64,
    )
