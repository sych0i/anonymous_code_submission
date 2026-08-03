"""Evaluation metrics for classifier-conditioned MNIST generation.

The primary metric follows the Binary-MNIST evaluation in VISTA: a sample is
valid when its target-class log probability is greater than ``-0.1``, and
valid samples are grouped by foreground Jaccard similarity.  The paper states
that a similarity of 0.85 is already a duplicate, so the comparison here is
inclusive.

The module also provides a soft-Jaccard distance for grayscale images.  It is
threshold-free and reduces exactly to ordinary Jaccard distance for binary
images.
"""

from __future__ import annotations

import math
from typing import Any

import torch


__all__ = [
    "compute_vista_metrics",
    "count_jaccard_components",
    "hard_jaccard_similarity_matrix",
    "normalize_mnist_samples",
    "soft_jaccard_diversity",
]


def _as_cpu_float_tensor(value: Any, name: str) -> torch.Tensor:
    """Convert an array-like value to a detached CPU float64 tensor."""
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor-like") from exc
    return tensor.detach().to(device="cpu", dtype=torch.float64)


def _validate_image_batch(images: torch.Tensor, name: str) -> None:
    if images.ndim < 2:
        raise ValueError(
            f"{name} must have shape (N, ...pixels), got {tuple(images.shape)}"
        )
    if math.prod(images.shape[1:]) == 0:
        raise ValueError(f"{name} must contain at least one pixel per sample")


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return value


def normalize_mnist_samples(
    samples: Any,
    *,
    sample_range: str = "minus_one_to_one",
) -> torch.Tensor:
    """Return detached MNIST samples in the closed interval ``[0, 1]``.

    Args:
        samples: Tensor-like batch with shape ``(N, ...pixels)``.  Both
            ``(N, H, W)`` and ``(N, C, H, W)`` are supported.
        sample_range: ``"minus_one_to_one"`` for diffusion outputs or
            ``"zero_to_one"`` for already-normalized images.

    Values are clipped after conversion because the diffusion sampler can
    produce small numerical excursions outside its nominal range.
    """
    images = _as_cpu_float_tensor(samples, "samples")
    _validate_image_batch(images, "samples")
    if images.numel() and not bool(torch.isfinite(images).all()):
        raise ValueError("samples must contain only finite values")

    if sample_range == "minus_one_to_one":
        images = (images + 1.0) / 2.0
    elif sample_range != "zero_to_one":
        raise ValueError(
            "sample_range must be 'minus_one_to_one' or 'zero_to_one'"
        )
    return images.clamp(0.0, 1.0)


def _as_binary_image_batch(binary_images: Any) -> torch.Tensor:
    images = torch.as_tensor(binary_images).detach().to(device="cpu")
    _validate_image_batch(images, "binary_images")
    if images.dtype == torch.bool:
        return images

    is_binary = (images == 0) | (images == 1)
    if images.numel() and not bool(is_binary.all()):
        raise ValueError("binary_images must contain only 0/1 or boolean values")
    return images.to(dtype=torch.bool)


def hard_jaccard_similarity_matrix(binary_images: Any) -> torch.Tensor:
    """Compute the pairwise foreground Jaccard similarity matrix.

    Two empty foreground masks have similarity one.  The returned tensor is a
    CPU float64 tensor with shape ``(N, N)``.
    """
    images = _as_binary_image_batch(binary_images)
    pixel_count = math.prod(images.shape[1:])
    flat = images.reshape(images.shape[0], pixel_count).to(dtype=torch.float64)

    intersection = flat @ flat.T
    areas = flat.sum(dim=1)
    union = areas[:, None] + areas[None, :] - intersection

    similarity = torch.ones_like(union)
    nonempty = union > 0
    similarity[nonempty] = intersection[nonempty] / union[nonempty]
    return similarity


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size
        self.count = size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        self.count -= 1


def count_jaccard_components(
    binary_images: Any,
    *,
    duplicate_threshold: float = 0.85,
) -> int:
    """Count connected components in the inclusive duplicate graph.

    Each image is a vertex and an undirected edge is added when pairwise
    Jaccard similarity is ``>= duplicate_threshold``.  The paper does not
    specify whether duplicate relations are transitively closed.  This
    continuous-domain adaptation uses deterministic connected components and
    honors the paper's inclusive threshold.
    """
    threshold = _validate_probability(
        duplicate_threshold, "duplicate_threshold"
    )
    similarities = hard_jaccard_similarity_matrix(binary_images)
    count = similarities.shape[0]
    if count < 2:
        return int(count)

    components = _DisjointSet(count)
    for left in range(count - 1):
        duplicate_right = torch.nonzero(
            similarities[left, left + 1 :] >= threshold,
            as_tuple=False,
        ).flatten()
        for offset in duplicate_right.tolist():
            components.union(left, left + 1 + int(offset))
    return int(components.count)


def _as_unit_image_batch(unit_images: Any) -> torch.Tensor:
    images = _as_cpu_float_tensor(unit_images, "unit_images")
    _validate_image_batch(images, "unit_images")
    if images.numel() and not bool(torch.isfinite(images).all()):
        raise ValueError("unit_images must contain only finite values")
    if images.numel() and bool(((images < 0.0) | (images > 1.0)).any()):
        raise ValueError("unit_images must lie in [0, 1]")
    return images


def soft_jaccard_diversity(unit_images: Any) -> float:
    """Return mean pairwise soft-Jaccard distance.

    For nonnegative images ``a`` and ``b``, soft-Jaccard similarity is

    ``sum(min(a, b)) / sum(max(a, b))``.

    Two all-zero images have similarity one.  Fewer than two samples have no
    pairwise diversity, so this function returns ``NaN`` in that case.
    """
    images = _as_unit_image_batch(unit_images)
    count = images.shape[0]
    if count < 2:
        return math.nan

    flat = images.reshape(count, -1)
    distance_sum = 0.0
    pair_count = 0

    # Iterate over one left image at a time to avoid materializing an
    # O(N^2 * pixels) broadcast tensor.
    for left in range(count - 1):
        right = flat[left + 1 :]
        left_expanded = flat[left].expand_as(right)
        numerator = torch.minimum(left_expanded, right).sum(dim=1)
        denominator = torch.maximum(left_expanded, right).sum(dim=1)

        similarity = torch.ones_like(denominator)
        nonempty = denominator > 0
        similarity[nonempty] = numerator[nonempty] / denominator[nonempty]
        distances = (1.0 - similarity).clamp(0.0, 1.0)

        distance_sum += float(distances.sum().item())
        pair_count += int(distances.numel())

    return float(distance_sum / pair_count)


def _select_target_log_probs(
    classifier_log_probs: Any,
    *,
    num_samples: int,
    target_label: int | None,
) -> torch.Tensor:
    log_probs = _as_cpu_float_tensor(
        classifier_log_probs, "classifier_log_probs"
    )
    if log_probs.numel() and bool(
        (torch.isnan(log_probs) | torch.isposinf(log_probs)).any()
    ):
        raise ValueError(
            "classifier_log_probs must not contain NaN or positive infinity"
        )

    if log_probs.ndim == 1:
        if log_probs.shape[0] != num_samples:
            raise ValueError(
                "classifier_log_probs and samples must have the same batch size"
            )
        return log_probs

    if log_probs.ndim != 2:
        raise ValueError(
            "classifier_log_probs must have shape (N,) or (N, num_classes)"
        )
    if log_probs.shape[0] != num_samples:
        raise ValueError(
            "classifier_log_probs and samples must have the same batch size"
        )
    if target_label is None:
        raise ValueError(
            "target_label is required for a full (N, num_classes) log-prob matrix"
        )
    if isinstance(target_label, bool) or not isinstance(target_label, int):
        raise TypeError("target_label must be an integer")
    if not 0 <= target_label < log_probs.shape[1]:
        raise ValueError(
            f"target_label {target_label} is outside [0, {log_probs.shape[1]})"
        )
    return log_probs[:, target_label]


def compute_vista_metrics(
    samples: Any,
    classifier_log_probs: Any,
    target_label: int | None = None,
    *,
    sample_range: str = "minus_one_to_one",
    valid_logprob_threshold: float = -0.1,
    foreground_threshold: float = 0.5,
    duplicate_threshold: float = 0.85,
) -> dict[str, int | float]:
    """Compute paper-faithful validity/uniqueness and grayscale diversity.

    ``classifier_log_probs`` may contain either target-class log probabilities
    with shape ``(N,)`` or the original-scale full classifier log-probability
    matrix with shape ``(N, num_classes)``.  In the latter case,
    ``target_label`` is required.  Validity uses a strict ``>`` comparison,
    matching the paper's reward criterion.

    All returned values are native Python ``int``/``float`` objects.  Metrics
    with a zero denominator, and pairwise diversity with fewer than two
    relevant samples, are represented by ``float("nan")``.
    """
    foreground_threshold = _validate_probability(
        foreground_threshold, "foreground_threshold"
    )
    duplicate_threshold = _validate_probability(
        duplicate_threshold, "duplicate_threshold"
    )
    valid_logprob_threshold = float(valid_logprob_threshold)
    if math.isnan(valid_logprob_threshold):
        raise ValueError("valid_logprob_threshold must not be NaN")

    unit_images = normalize_mnist_samples(samples, sample_range=sample_range)
    num_samples = int(unit_images.shape[0])
    target_log_probs = _select_target_log_probs(
        classifier_log_probs,
        num_samples=num_samples,
        target_label=target_label,
    )

    valid_mask = target_log_probs > valid_logprob_threshold
    valid_count = int(valid_mask.sum().item())

    binary_images = unit_images >= foreground_threshold
    unique_valid_count = count_jaccard_components(
        binary_images[valid_mask],
        duplicate_threshold=duplicate_threshold,
    )

    if num_samples:
        valid_rate = float(valid_count / num_samples)
        unique_valid_rate = float(unique_valid_count / num_samples)
    else:
        valid_rate = math.nan
        unique_valid_rate = math.nan

    conditional_unique_fraction = (
        float(unique_valid_count / valid_count) if valid_count else math.nan
    )

    return {
        "num_samples": num_samples,
        "valid_count": valid_count,
        "valid_rate": valid_rate,
        "unique_valid_count": unique_valid_count,
        "conditional_unique_fraction": conditional_unique_fraction,
        "unique_valid_rate": unique_valid_rate,
        "soft_jaccard_diversity_all": soft_jaccard_diversity(unit_images),
        "soft_jaccard_diversity_valid": soft_jaccard_diversity(
            unit_images[valid_mask]
        ),
    }
