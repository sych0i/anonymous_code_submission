import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)                  # Python random module
    np.random.seed(seed)               # NumPy random seed
    torch.manual_seed(seed)            # PyTorch CPU seed
    torch.cuda.manual_seed(seed)       # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)   # All GPUs seed (if using multi-GPU)

    # # Ensure deterministic behavior
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False