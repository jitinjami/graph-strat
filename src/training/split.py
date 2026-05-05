"""K-fold split strategies for transductive node classification.

Strategies
----------
class_wise  — StratifiedKFold, class-proportional folds.
random      — KFold, purely random folds.
hpstrat — HomophilyKFold, homophily-bin-first stratification.

Val set = folds[(fold + 1) % k] — rotating round-robin.
"""

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

from src.training.hp_split import HomophilyKFold


def get_all_folds(
    data,
    y: torch.Tensor,
    strategy: str = "class_wise",
    k: int = 4,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return k (train_mask, val_mask, test_mask) tuples — one per fold.

    Splits are computed once; calling this function runs any expensive
    operations (e.g. homophily computation for hpstrat) only once.

    Args:
        data:     PyG Data object. Required by hpstrat for edge_index.
        y:        1-D integer label tensor of length N.
        strategy: One of 'class_wise', 'random', 'hpstrat'.
        k:        Number of folds (>= 2).
        seed:     RNG seed.

    Returns:
        List of k tuples (train_mask, val_mask, test_mask), each a boolean
        tensor of shape (N,).
    """
    if k < 2:
        raise ValueError(f"k={k} must be >= 2.")

    n      = y.size(0)
    labels = y.cpu().numpy()
    valid  = labels >= 0
    valid_idx = np.where(valid)[0]

    if strategy == "class_wise":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(splitter.split(valid_idx, labels[valid_idx]))
    elif strategy == "random":
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(splitter.split(valid_idx))
    elif strategy == "hpstrat":
        splitter = HomophilyKFold(n_splits=k, seed=seed)
        splitter._full_labels = labels
        folds = list(splitter.split(
            valid_idx,
            labels[valid_idx],
            data.edge_index.cpu().numpy(),
        ))
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Valid: 'class_wise', 'random', 'hpstrat'."
        )

    masks = []
    for fold in range(k):
        test_idx = valid_idx[folds[fold][1]]
        val_idx  = valid_idx[folds[(fold + 1) % k][1]]

        exclude = np.zeros(n, dtype=bool)
        exclude[test_idx] = True
        exclude[val_idx]  = True
        train_idx = np.where(valid & ~exclude)[0]

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True

        masks.append((train_mask, val_mask, test_mask))

    return masks
