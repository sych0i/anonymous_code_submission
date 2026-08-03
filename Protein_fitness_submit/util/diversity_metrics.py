"""Diversity-aware reward metrics.

Plain mean/max reward over a batch of samples can be dominated by many
near-identical copies of a single good sequence. `num_unique_high_reward`
instead counts how many *distinct* solutions clear a reward bar, where
"distinct" is defined by a similarity threshold rather than exact string
match (`len(set(combos))`, used elsewhere in this repo, only catches exact
duplicates).
"""
import numpy as np


def hamming_similarity(a, b):
    """Fraction of matching positions between two equal-length strings/sequences."""
    if len(a) != len(b):
        raise ValueError(f"hamming_similarity requires equal-length inputs, got {len(a)} and {len(b)}")
    if len(a) == 0:
        return 1.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cluster_by_similarity(seqs, sim_threshold, similarity_fn=hamming_similarity):
    """Union-find clustering: seqs[i] and seqs[j] land in the same cluster iff
    similarity_fn(seqs[i], seqs[j]) >= sim_threshold.

    Returns a list of clusters, each a list of indices into `seqs`.
    """
    n = len(seqs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_fn(seqs[i], seqs[j]) >= sim_threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def num_unique_high_reward(seqs, rewards, reward_threshold, sim_threshold, similarity_fn=hamming_similarity):
    """Count distinct high-reward solutions.

    Filters to samples with reward >= reward_threshold, then clusters the
    survivors by pairwise similarity (>= sim_threshold counts as "the same
    sample"), and returns the number of resulting clusters. This answers
    "how many genuinely different good solutions did we find", as opposed to
    mean/max reward (which don't penalize mode collapse) or exact-match
    uniqueness (which treats a single point mutation as a different sample).

    Args:
        seqs: sequence of samples (e.g. residue-combo strings), same length as rewards.
        rewards: array-like of scalar rewards, same length as seqs.
        reward_threshold: minimum reward for a sample to count at all.
        sim_threshold: similarity_fn value at or above which two samples are
            considered the same underlying solution.
        similarity_fn: (a, b) -> float in [0, 1], higher = more similar.
            Defaults to normalized Hamming similarity (fixed-length sequences).

    Returns:
        int: number of distinct clusters among the high-reward survivors.
    """
    rewards = np.asarray(rewards)
    keep = np.where(rewards >= reward_threshold)[0]
    if len(keep) == 0:
        return 0
    kept_seqs = [seqs[i] for i in keep]
    clusters = cluster_by_similarity(kept_seqs, sim_threshold, similarity_fn)
    return len(clusters)
