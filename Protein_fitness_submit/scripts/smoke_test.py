"""Fast GPU check for both packaged checkpoints, the oracle, and SMC-base."""

import argparse
import os
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.chdir(PACKAGE_ROOT)

from problem.protein_reward import ProteinOracleReward
from sampling.smc import SMC_Base
from util.seed import set_seed

try:
    from .experiment_spec import BACKBONES, ESS_THRESHOLD, PARTIAL_RESAMPLE
except ImportError:
    from experiment_spec import BACKBONES, ESS_THRESHOLD, PARTIAL_RESAMPLE


def run(backbone, device):
    data_config = OmegaConf.load(PACKAGE_ROOT / "configs" / "data" / "TrpB.yaml")
    model_config = OmegaConf.load(PACKAGE_ROOT / "configs" / "model" / f"{backbone}.yaml")
    set_seed(42)
    net = instantiate(
        model_config.model,
        model_name=f"{backbone}/TrpB",
        seq_len=data_config.seq_len,
        num_steps=4,
        device=device,
        _recursive_=False,
    )
    oracle = ProteinOracleReward(data_config=data_config, device=device)
    algo = SMC_Base(
        net=net,
        forward_op=oracle,
        data_config=data_config,
        alpha=BACKBONES[backbone]["alpha"],
        num_rollout_samples=1,
        ess_threshold=ESS_THRESHOLD,
        partial_resample=PARTIAL_RESAMPLE,
        device=device,
    )
    set_seed(123)
    _, sequences = algo.inference(
        num_samples=1, verbose=False, detokenize=True, guided_set={1, 3})
    combo = "".join(sequences[0][index] for index in algo.residues)
    reward = float(oracle([combo])[0])
    print(f"{backbone}: combo={combo} reward={reward:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", choices=sorted(BACKBONES), default=sorted(BACKBONES))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the model smoke test requires an NVIDIA CUDA GPU")
    device = torch.device("cuda")
    for backbone in args.backbones:
        run(backbone, device)
    print("smoke test: OK")


if __name__ == "__main__":
    main()
