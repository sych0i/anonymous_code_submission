"""Minimal protein tokenizer used by the supplied MDLM and UDLM checkpoints."""

import numpy as np
import torch


class ProteinTokenizer:
    """Map the checkpoints' fixed protein alphabet to and from token IDs."""

    ALPHABET = "ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!"

    def __init__(self, sequences=True):
        self.alphabet = list(self.ALPHABET)
        self.a_to_i = {token: index for index, token in enumerate(self.alphabet)}
        self.i_to_a = np.asarray(self.alphabet)
        self.sequences = sequences
        self.K = 26 if sequences else 27

    @property
    def mask_id(self):
        return self.a_to_i["#"]

    @property
    def pad_id(self):
        return self.a_to_i["!"]

    def tokenize(self, sequence):
        # Experiment configs store each full sequence as a one-element list.
        value = sequence[0]
        return np.asarray([self.a_to_i[token] for token in value])

    def untokenize(self, tokens):
        if torch.is_tensor(tokens):
            tokens = tokens.detach().cpu().tolist()
        return "".join(self.i_to_a[int(token)] for token in tokens)
