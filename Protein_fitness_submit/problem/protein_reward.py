import os

import numpy as np
import torch

from .base import BaseOperator
from oracle.model import OracleModel, HIDDEN_DIM, DROPOUT

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

# Same (cutoff, rate) pairs used in oracle/inference_oracle.py::inference_oracle
HAMMING_PENALTY = {
    "CreiLOV": (70, 0.99),
    "TrpB": (233, 0.99),
    "GB1": (33, 0.99),
}


def _onehot_combos(combos, wt_len):
    encodings = np.zeros((len(combos), wt_len, len(STANDARD_AAS)), dtype=np.float32)
    for i, combo in enumerate(combos):
        for j, aa in enumerate(combo):
            encodings[i, j, STANDARD_AAS.index(aa)] = 1.0
    return torch.from_numpy(encodings.reshape(len(combos), -1))


def _hamming_to_wt(combos, wt_combo):
    return np.array([sum(c1 != c2 for c1, c2 in zip(combo, wt_combo)) for combo in combos])


def _apply_hamming_penalty(preds, combos, wt_combo, protein):
    cutoff, rate = HAMMING_PENALTY[protein]
    hamming = _hamming_to_wt(combos, wt_combo)
    penalty = torch.tensor(
        np.where(hamming <= cutoff, 1.0, rate ** (hamming - cutoff)),
        dtype=preds.dtype, device=preds.device,
    )
    return preds * penalty


def _soft_reward_metadata(data_config):
    '''Build the mapping from full diffusion states to the reward MLP's flat 20-AA input.

    ``data_config.residues`` is specified with biological (1-based) coordinates, whereas the
    diffusion state is indexed from zero.  The standard amino acids are looked up by symbol in
    the model alphabet rather than assumed to occupy its first 20 channels.
    '''
    alphabet = tuple(data_config.alphabet)
    seq_len = int(data_config.seq_len)
    full_seq = str(data_config.full_seq)
    if len(full_seq) != seq_len:
        raise ValueError(
            f"full_seq has length {len(full_seq)}, but data_config.seq_len is {seq_len}")

    invalid_standard_aas = [aa for aa in STANDARD_AAS if alphabet.count(aa) != 1]
    if invalid_standard_aas:
        raise ValueError(
            "data_config.alphabet must contain each standard amino acid exactly once; "
            f"invalid entries: {invalid_standard_aas}")
    standard_aa_token_indices = tuple(alphabet.index(aa) for aa in STANDARD_AAS)

    if data_config.residues is None:
        residue_indices = tuple(range(seq_len))
    else:
        residue_indices = tuple(int(residue) - 1 for residue in data_config.residues)
    invalid_residues = [index + 1 for index in residue_indices
                        if index < 0 or index >= seq_len]
    if invalid_residues:
        raise ValueError(
            f"data_config.residues contains out-of-range 1-based positions: {invalid_residues}")

    wt_combo = ''.join(full_seq[index] for index in residue_indices)
    missing_wt_tokens = sorted(set(wt_combo).difference(alphabet))
    if missing_wt_tokens:
        raise ValueError(
            "wild-type residues are missing from data_config.alphabet: "
            f"{missing_wt_tokens}")
    wt_token_indices = tuple(alphabet.index(aa) for aa in wt_combo)
    return (alphabet, seq_len, residue_indices, standard_aa_token_indices,
            wt_token_indices, wt_combo)


def _freeze_reward_model(model):
    '''Freeze reward parameters while retaining gradients with respect to model inputs.'''
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _apply_soft_hamming_penalty(preds, token_probs, residue_indices, wt_token_indices,
                                protein):
    '''Apply the hard-sequence penalty without differentiating through argmax.'''
    positions = torch.as_tensor(
        residue_indices, device=token_probs.device, dtype=torch.long)
    wt_tokens = torch.as_tensor(
        wt_token_indices, device=token_probs.device, dtype=torch.long)
    hard_tokens = token_probs.detach().index_select(1, positions).argmax(dim=-1)
    hamming = (hard_tokens != wt_tokens.unsqueeze(0)).sum(dim=-1).to(preds.device)

    cutoff, rate = HAMMING_PENALTY[protein]
    # The legacy string path computes the power in NumPy float64 and then casts to the prediction
    # dtype.  Mirror that precision so hard one-hot inputs agree down to the final cast.
    excess = (hamming - cutoff).clamp_min(0).to(torch.float64)
    penalty = torch.pow(
        torch.tensor(rate, device=preds.device, dtype=torch.float64), excess).to(preds.dtype)
    return preds * penalty


def _score_soft_models(token_probs, models, alphabet, seq_len, residue_indices,
                       standard_aa_token_indices, wt_token_indices, protein,
                       impose_penalty):
    '''Score a differentiable full-vocabulary relaxation with a reward-model ensemble.'''
    if not isinstance(token_probs, torch.Tensor):
        raise TypeError(f"token_probs must be a torch.Tensor, got {type(token_probs).__name__}")
    if token_probs.ndim != 3:
        raise ValueError(
            f"token_probs must have shape (B, L, V), got {tuple(token_probs.shape)}")
    if token_probs.shape[1] != seq_len:
        raise ValueError(
            "token_probs must cover the full sequence: "
            f"expected L={seq_len}, got L={token_probs.shape[1]}")
    if token_probs.shape[2] != len(alphabet):
        raise ValueError(
            "token_probs must cover the full model vocabulary: "
            f"expected V={len(alphabet)}, got V={token_probs.shape[2]}")
    if not token_probs.is_floating_point():
        raise TypeError("token_probs must have a floating-point dtype")
    if not models:
        raise RuntimeError("cannot score without at least one loaded reward model")

    positions = torch.as_tensor(
        residue_indices, device=token_probs.device, dtype=torch.long)
    aa_channels = torch.as_tensor(
        standard_aa_token_indices, device=token_probs.device, dtype=torch.long)
    x = token_probs.index_select(1, positions).index_select(2, aa_channels)

    first_parameter = next(models[0].parameters())
    x = x.reshape(x.shape[0], -1).to(
        device=first_parameter.device, dtype=first_parameter.dtype)
    preds = torch.stack([model(x).squeeze(-1) for model in models], dim=0).mean(dim=0)

    if impose_penalty:
        preds = _apply_soft_hamming_penalty(
            preds, token_probs, residue_indices, wt_token_indices, protein)
    return preds.clamp(min=0.0)


class ProteinOracleReward(BaseOperator):
    '''
    Reward function r(x0) = predicted fitness of a clean protein sequence, used as the
    reward signal *inside* SMC guidance (i.e. the "r" in R(x0) = exp(r(x0)/alpha)).

    This wraps the same oracle ensemble used by oracle/inference_oracle.py, but loads the
    ensemble checkpoints once and keeps them resident on `device`, since SMC needs to call
    the reward many times (once per rollout, at every guided timestep) and reloading state
    dicts from disk on every call would dominate runtime.
    '''

    def __init__(self, data_config, model_path=None, device='cuda', impose_penalty=True):
        self.device = device
        self.protein = data_config.name
        self.model_path = model_path if model_path is not None else data_config.oracle_model_path
        self.impose_penalty = impose_penalty

        (self._model_alphabet, self._seq_len, self._residue_indices,
         self._standard_aa_token_indices, self._wt_token_indices,
         self.wt_combo) = _soft_reward_metadata(data_config)

        files = sorted(os.listdir(self.model_path))
        input_dim = len(self.wt_combo) * len(STANDARD_AAS)
        self.models = []
        for f in files:
            model = OracleModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT)
            model.load_state_dict(torch.load(os.path.join(self.model_path, f), map_location=device))
            model.to(device)
            _freeze_reward_model(model)
            self.models.append(model)

    @torch.no_grad()
    def __call__(self, combos, **kwargs):
        '''
        combos: list[str] of amino acid combos at the mutated residues (already restricted
            to the standard 20 AAs upstream).
        Returns: torch.Tensor of shape (len(combos),) with the ensemble-averaged, clipped
            (and optionally hamming-penalized) predicted fitness -- matches the semantics of
            oracle/inference_oracle.py::inference_oracle.
        '''
        if len(combos) == 0:
            return torch.zeros(0, device=self.device)

        x = _onehot_combos(combos, len(self.wt_combo)).to(self.device)
        preds = torch.stack([m(x).squeeze(-1) for m in self.models], dim=0).mean(dim=0)

        if self.impose_penalty:
            preds = _apply_hamming_penalty(preds, combos, self.wt_combo, self.protein)

        return preds.clamp(min=0.0)

    def score_soft(self, token_probs):
        '''Differentiably score full-vocabulary probabilities of shape ``(B, L, V)``.

        Only configured mutable positions and the 20 standard-AA channels are passed to the
        oracle MLP.  The standard channels are deliberately not renormalized: this is the direct
        continuous relaxation of the one-hot input used by :meth:`__call__`.
        '''
        return _score_soft_models(
            token_probs, self.models, self._model_alphabet, self._seq_len,
            self._residue_indices, self._standard_aa_token_indices,
            self._wt_token_indices, self.protein, self.impose_penalty)

    def loss(self, inputs, y=None, **kwargs):
        return -self(inputs, **kwargs)
