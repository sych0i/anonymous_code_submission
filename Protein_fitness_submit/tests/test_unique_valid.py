import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.diversity_metrics import num_unique_high_reward


def test_unique_valid_filters_reward_and_clusters_one_mismatch():
    sequences = [
        "AAAAAAAAAAAAAAA",
        "CAAAAAAAAAAAAAA",  # one mismatch from the first: same component
        "CCAAAAAAAAAAAAA",  # linked transitively through the second
        "CCCCCCCCCCCCCCC",
        "GGGGGGGGGGGGGGG",  # below the reward threshold
    ]
    rewards = [0.9, 0.8, 0.7, 0.6, 0.49]
    count = num_unique_high_reward(
        sequences,
        rewards,
        reward_threshold=0.5,
        sim_threshold=1 - 1 / 15,
    )
    assert count == 2


def test_duplicate_sequences_count_once():
    sequences = ["ACDEFGHIKLMNPQR"] * 4
    rewards = [0.5, 0.6, 0.7, 0.8]
    assert num_unique_high_reward(sequences, rewards, 0.5, 1 - 1 / 15) == 1
