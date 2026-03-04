"""
Evaluation metrics, parameter utilities, and training helpers.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE


def flatten_params(model: nn.Module) -> torch.Tensor:
    """
    Flattens all model parameters into a single 1D tensor.

    Args:
        model: PyTorch model

    Returns:
        1D tensor with all parameters concatenated
    """
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def flatten_grads(model: nn.Module) -> torch.Tensor:
    """
    Flattens all model gradients into a single 1D tensor.
    Parameters with no gradient are replaced by zeros.

    Args:
        model: PyTorch model

    Returns:
        1D tensor with all gradients concatenated
    """

    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.detach().view(-1))
    return torch.cat(grads)


def load_flat_params_(model: nn.Module, flat: torch.Tensor) -> None:
    """
    Loads a flat parameter tensor back into the model in-place.

    Args:
        model: PyTorch model to update
        flat:  1D tensor with new parameter values
    """

    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].view_as(p))
        offset += n



@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, max_batches: int = 20) -> float:
    """
    Computes the average cross-entropy loss over a dataloader.

    Args:
        model:       model to evaluate
        loader:      dataloader to evaluate on
        max_batches: maximum number of batches to use

    Returns:
        Mean cross-entropy loss
    """

    model.eval()
    total = 0.0
    n = 0
    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, max_batches: int = 80) -> float:
    """
    Computes classification accuracy over a dataloader.

    Args:
        model:       model to evaluate
        loader:      dataloader to evaluate on
        max_batches: maximum number of batches to use

    Returns:
        Accuracy as a float in [0, 1]
    """

    model.eval()
    correct = 0
    total = 0
    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)



@torch.no_grad()
def probing_loss(model: nn.Module, loader: DataLoader, batches: int = 1) -> float:
    """
    Computes the average cross-entropy loss over the first N batches.

    Args:
        model:   model to evaluate
        loader:  dataloader to probe
        batches: number of batches to use

    Returns:
        Mean cross-entropy loss
    """

    model.eval()
    tot = 0.0
    n = 0
    for b, (x, y) in enumerate(loader):
        if b >= batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        tot += loss.item()
        n += 1
    return tot / max(1, n)


@torch.no_grad()
def probing_loss_random_offset(
    model: nn.Module,
    loader: DataLoader,
    batches: int = 1,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """
    Computes probing loss starting from a random batch offset.

    Instead of always evaluating the first batches, starts from a random
    position in the loader to reduce evaluation bias across rounds.

    Args:
        model:   model to evaluate
        loader:  dataloader to probe
        batches: number of batches to use
        rng:     random state for offset sampling

    Returns:
        Mean cross-entropy loss over the selected batches
    """

    model.eval()
    if rng is None:
        rng = np.random.RandomState(0)

    try:
        n_total_batches = len(loader)
    except TypeError:
        n_total_batches = 0

    if n_total_batches <= 0:
        return probing_loss(model, loader, batches=batches)

    b = max(1, int(batches))
    max_start = max(0, n_total_batches - b)
    start = int(rng.randint(0, max_start + 1))

    tot = 0.0
    n = 0
    for bi, (x, y) in enumerate(loader):
        if bi < start:
            continue
        if bi >= start + b:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        tot += loss.item()
        n += 1

    return tot / max(1, n)



def gini_coefficient(x: np.ndarray) -> float:
    """
    Computes the Gini coefficient of an array, used to measure
    selection imbalance across clients.

    A value of 0 means all clients are selected equally.
    A value of 1 means only one client is ever selected.

    Args:
        x: array of non-negative values (e.g. per-client selection counts)

    Returns:
        Gini coefficient in [0, 1]
    """

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    g = (n + 1.0 - 2.0 * np.sum(cum) / cum[-1]) / n
    return float(g)



def windowed_reward(loss_history: list, new_loss: float, W: int = 5) -> float:
    """
    Computes the normalized reward based on validation loss improvement
    over a sliding window of previous rounds.

    Defined as (mean_window_loss - new_loss) / (mean_window_loss + eps),
    so positive values indicate improvement and negative values indicate degradation.

    Args:
        loss_history: list of previous validation losses
        new_loss:     validation loss after the current round
        W:            window size

    Returns:
        Normalized reward scalar
    """

    if len(loss_history) == 0:
        base = new_loss
    else:
        w = min(W, len(loss_history))
        base = float(np.mean(loss_history[-w:]))
    raw = float(base - new_loss)
    denom = float(abs(base) + 1e-6)
    return raw / denom


def dynamic_batch_size(buf_n: int, base: int = 64, max_bs: int = 256, ratio: int = 4) -> int:
    """
    Computes a dynamic batch size that grows with the replay buffer size.

    Doubles the batch size each time the buffer has at least ratio times
    the current batch size, up to max_bs.

    Args:
        buf_n:  current number of transitions in the buffer
        base:   minimum batch size
        max_bs: maximum batch size
        ratio:  buffer-to-batch ratio threshold for doubling

    Returns:
        Batch size to use for the current training step
    """
    
    bs = int(base)
    while (bs < max_bs) and (buf_n >= ratio * bs):
        bs *= 2
    return min(bs, max_bs)
