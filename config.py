"""
Global configuration: random seeds, device selection, and utility functions.
"""
import random

import numpy as np
import torch

SEED = 2048
"""Global random seed for reproducibility across all libraries."""

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""Device used for training: 'cuda' if available, otherwise 'cpu'."""


def seed_worker(worker_id: int):
    """
    Worker initialization function for DataLoader reproducibility.

    Sets the random seed for each worker process based on the global SEED,
    ensuring deterministic data loading across runs.

    Args:
        worker_id: index of the DataLoader worker process
    """
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def log_step(msg: str):
    """
    Prints a message with immediate flush to stdout.

    Args:
        msg: message to print
    """
    print(msg, flush=True)
    print(msg, flush=True)
