"""
Dataset utilities: label flipping attack and non-IID data partitioning.
"""
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset, Subset


class SwitchableTargetedLabelFlipSubset(Dataset):
    """
    Dataset wrapper that simulates a malicious client performing targeted label flipping attacks.

    For each sample, a random value is pre-computed at initialization to decide
    whether that sample will be flipped. At runtime, a sample is flipped if the
    attack is enabled and its random value is below attack_rate — this makes the
    attack deterministic and reproducible across runs.

    The flipping follows a fixed class mapping (target_map), where each class is
    mapped to a specific target class. By default, visually similar classes are
    swapped (e.g. airplane <-> ship, cat <-> dog).

    Args:
        base_ds:          base PyTorch dataset
        indices:          sample indices to use from base_ds
        n_classes:        number of classes
        seed:             random seed for reproducibility
        enabled:          whether the attack is active
        attack_rate:      fraction of samples to flip (0.0 to 1.0)
        target_map:       dict mapping original class to target class
        only_map_classes: if True, only flips classes present in target_map
    """

    def __init__(
        self,
        base_ds,
        indices,
        n_classes: int = 10,
        seed: int = 0,
        enabled: bool = False,
        attack_rate: float = 0.0,
        target_map: Optional[Dict[int, int]] = None,
        only_map_classes: bool = True,
    ):
        self.base_ds = base_ds
        self.indices = list(indices)
        self.n_classes = int(n_classes)
        self.enabled = bool(enabled)
        self.attack_rate = float(attack_rate)
        self.only_map_classes = bool(only_map_classes)

        if target_map is None:
            target_map = {
                0: 8,  # airplane -> ship
                8: 0,  # ship -> airplane
                1: 9,  # automobile -> truck
                9: 1,  # truck -> automobile
                3: 5,  # cat -> dog
                5: 3,  # dog -> cat
                4: 7,  # deer -> horse
                7: 4,  # horse -> deer
                2: 6,  # bird -> frog
                6: 2,  # frog -> bird
            }
        self.target_map = {int(k): int(v) for k, v in target_map.items()}

        rng = np.random.RandomState(seed)
        self.u = rng.rand(len(self.indices)).astype(np.float32)

        self.flipped_label = np.zeros(len(self.indices), dtype=np.int64)
        for i, idx in enumerate(self.indices):
            _, y = self.base_ds[idx]
            y = int(y)

            if y in self.target_map:
                y_new = int(self.target_map[y])
            else:
                if self.only_map_classes:
                    y_new = y
                else:
                    y_new = rng.randint(0, self.n_classes - 1)
                    if y_new >= y:
                        y_new += 1

            y_new = int(np.clip(y_new, 0, self.n_classes - 1))
            if y_new == y:
                y_new = (y + 1) % self.n_classes

            self.flipped_label[i] = y_new

    def set_attack(self, enabled: bool, rate: float):
        """
        Enables or disables the attack at runtime.

        Args:
            enabled: whether to activate label flipping
            rate:    fraction of samples to flip
        """

        self.enabled = bool(enabled)
        self.attack_rate = float(rate)

    def __len__(self):
        """Returns the number of samples in the subset."""

        return len(self.indices)

    def __getitem__(self, i):
        """
        Returns sample i. If the attack is active and u[i] < attack_rate,
        the original label is replaced by the precomputed flipped label.

        Args:
            i: sample index

        Returns:
            Tuple (x, y) where y is the original or flipped label
        """

        x, y = self.base_ds[self.indices[i]]
        y = int(y)
        if self.enabled and (self.attack_rate > 0.0) and (self.u[i] < self.attack_rate):
            y = int(self.flipped_label[i])
        return x, y


def make_server_val_balanced(
    ds, per_class: int = 200, n_classes: int = 10, seed: int = 0
) -> List[int]:
    """
    Builds a balanced validation set for the server by sampling equally from each class.

    Args:
        ds:        base dataset to sample from
        per_class: number of samples per class
        n_classes: number of classes
        seed:      random seed for reproducibility

    Returns:
        List of indices forming the balanced validation set
    """

    rng = np.random.RandomState(seed)
    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx in range(len(ds)):
        _, y = ds[idx]
        label_to_idxs[int(y)].append(idx)

    val = []
    for y in range(n_classes):
        idxs = label_to_idxs[y]
        rng.shuffle(idxs)
        val.extend(idxs[:per_class])

    rng.shuffle(val)
    return val



def make_clients_dirichlet_indices(
    train_ds,
    n_clients: int = 50,
    alpha: float = 0.3,
    seed: int = 123,
    n_classes: int = 10,
) -> List[List[int]]:
    """
    Partitions the training dataset across clients using a Dirichlet distribution,
    simulating non-IID data heterogeneity.

    Lower values of alpha produce more heterogeneous distributions, where each
    client holds samples concentrated in fewer classes. Higher values approximate
    a uniform (IID) distribution.

    Args:
        train_ds:  base training dataset
        n_clients: number of federated clients
        alpha:     Dirichlet concentration parameter
        seed:      random seed for reproducibility
        n_classes: number of classes

    Returns:
        List of length n_clients, where each element is a list of sample indices
        assigned to that client
    """

    rng = np.random.RandomState(seed)

    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx in range(len(train_ds)):
        _, y = train_ds[idx]
        label_to_idxs[int(y)].append(idx)

    for y in range(n_classes):
        rng.shuffle(label_to_idxs[y])

    clients = [[] for _ in range(n_clients)]

    for y in range(n_classes):
        idxs = label_to_idxs[y]
        props = rng.dirichlet(alpha * np.ones(n_clients))
        counts = (props * len(idxs)).astype(int)

        diff = len(idxs) - counts.sum()
        if diff > 0:
            for j in rng.choice(n_clients, size=diff, replace=True):
                counts[j] += 1
        elif diff < 0:
            for j in rng.choice(np.where(counts > 0)[0], size=-diff, replace=True):
                counts[j] -= 1

        start = 0
        for cid in range(n_clients):
            c = counts[cid]
            if c > 0:
                clients[cid].extend(idxs[start: start + c])
                start += c

    for cid in range(n_clients):
        rng.shuffle(clients[cid])

    return clients
