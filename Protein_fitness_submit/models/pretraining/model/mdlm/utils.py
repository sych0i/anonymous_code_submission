"""Inference-time numerical diagnostics."""

import torch


def print_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(name, tensor)
