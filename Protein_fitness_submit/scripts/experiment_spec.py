"""Authoritative configuration for the Protein Fitness SMC-base tables."""

T = 128
SPARSE_T_PRIMES = (16, 32)
T_PRIMES = (*SPARSE_T_PRIMES, T)
N_PARTICLES = 32
N_RUNS = 20
N_ROLLOUTS = 3
N_WARMUP_RUNS = 3
WARMUP_SEED = 1000
ESS_THRESHOLD = 0.98
PARTIAL_RESAMPLE = True
REWARD_THRESHOLD = 0.5
SEQUENCE_LENGTH = 15
SIMILARITY_THRESHOLD = 1.0 - 1.0 / SEQUENCE_LENGTH

POLICIES = (
    "interval1",
    "interval2",
    "interval3",
    "interval4",
    "interval5",
    "top_v",
    "top_dv",
    "uniform",
    "vista",
)

DISPLAY_NAMES = {
    "interval1": "Interval-1",
    "interval2": "Interval-2",
    "interval3": "Interval-3",
    "interval4": "Interval-4",
    "interval5": "Interval-5",
    "top_v": "Top-V",
    "top_dv": "Top-dV",
    "uniform": "Uniform",
    "vista": "VISTA",
}

BACKBONES = {
    "mdlm": {
        "display_name": "MDLM",
        "alpha": 0.3,
        "evaluation_seed": 8000,
        "vista_k": None,
    },
    "udlm": {
        "display_name": "UDLM",
        "alpha": 0.2,
        "evaluation_seed": 5000,
        "vista_k": 1.0,
    },
}


def table_config(backbone, budget):
    """Return the portable metadata every metric/timing artifact must share."""
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone: {backbone}")
    if budget not in T_PRIMES:
        raise ValueError(f"T' must be one of {T_PRIMES}, got {budget}")
    model = BACKBONES[backbone]
    return {
        "backbone": backbone,
        "algo": "smc_base",
        "T": T,
        "budget": budget,
        "N": N_PARTICLES,
        "J": N_ROLLOUTS,
        "M": N_WARMUP_RUNS,
        "repeats": N_RUNS,
        "warmup_seed": WARMUP_SEED,
        "evaluation_seed": model["evaluation_seed"],
        "alpha": model["alpha"],
        "ess_threshold": ESS_THRESHOLD,
        "partial_resample": PARTIAL_RESAMPLE,
        "vista_k": model["vista_k"],
        "reward_threshold": REWARD_THRESHOLD,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "policies": ["full"] if budget == T else list(POLICIES),
    }
