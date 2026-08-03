from abc import ABC, abstractmethod
import torch

class GenerativeModel(ABC):
    def __init__(self):
        pass

    def score(self, x, t):
        pass

    def pred_mean(self, x, t):
        pass

    def get_start(self):
        pass

    def q_sample(self, x, t):
        pass