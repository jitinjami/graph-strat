"""Homophily-aware k-fold splitter for transductive node classification.

``hpstrat`` stratification: outer loop over homophily bins, inner loop
over class labels within each bin. Prioritises preserving the homophily
distribution across folds; class balance is best-effort.

Node homophily h(v) is the fraction of a node's neighbours that share its
class label (range [0, 1]).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection._split import _BaseKFold
from torch_geometric.utils import scatter

_N_BINS    = 10
_BIN_EDGES = np.linspace(0.0, 1.0, _N_BINS + 1)


def _compute_node_homophily(
    labels: np.ndarray,
    edge_index: np.ndarray,   # shape (2, E), numpy int64
) -> np.ndarray:
    """Return per-node homophily: fraction of same-label neighbours.

        h_v = |{w ∈ N(v) : y_w == y_v}| / |N(v)|

    Isolated nodes receive 0.0; nodes with label < 0 receive -1.0.
    """
    n = len(labels)
    src = torch.from_numpy(edge_index[0]).long()
    dst = torch.from_numpy(edge_index[1]).long()
    y   = torch.from_numpy(labels).long()

    same = (y[src] == y[dst]).double()
    hv   = scatter(same, dst, dim=0, dim_size=n, reduce="mean")

    out = hv.numpy().astype(np.float64)
    out[labels < 0] = -1.0
    return out


def _assign_bins(homophily: np.ndarray) -> np.ndarray:
    """Map h(v) values in [0, 1] to integer bin indices 0 … N_BINS-1."""
    bins = np.digitize(homophily, _BIN_EDGES[1:-1])
    bins = np.clip(bins, 0, _N_BINS - 1)
    bins[homophily < 0] = -1
    return bins.astype(np.int32)


class HomophilyKFold(_BaseKFold):
    """K-fold splitter stratified by homophily bin then class label.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    seed : int
        Random seed for within-stratum shuffling.
    """

    def __init__(self, n_splits: int = 10, seed: int = 42) -> None:
        super().__init__(n_splits=n_splits, shuffle=True, random_state=seed)
        self.seed = seed

    def split(self, X, y=None, edge_index=None):
        """Yield (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X          : global node indices of labeled nodes, shape (n_valid,)
        y          : class labels for those nodes, shape (n_valid,)
        edge_index : full graph edge index, shape (2, E), numpy int64
        """
        X   = np.asarray(X)
        y   = np.asarray(y)
        rng = np.random.default_rng(self.seed)

        homophily = _compute_node_homophily(self._full_labels, edge_index)
        bins      = _assign_bins(homophily[X])

        assignments = self._assign_folds(y, bins, len(X), rng)

        for fold in range(self.n_splits):
            test  = assignments == fold
            train = ~test
            yield np.where(train)[0], np.where(test)[0]

    def _assign_folds(self, labels, bins, n, rng):
        assignments = np.full(n, -1, dtype=np.int32)
        fold_sizes  = np.zeros(self.n_splits, dtype=np.int64)
        k = self.n_splits

        for bin_id in range(_N_BINS):
            outer = bins == bin_id
            if not np.any(outer):
                continue
            for cls in np.unique(labels[outer & (labels >= 0)]):
                stratum = rng.permutation(np.where(outer & (labels == cls))[0])
                start   = int(np.argmin(fold_sizes))
                for rank, idx in enumerate(stratum):
                    f = (start + rank) % k
                    assignments[idx] = f
                    fold_sizes[f] += 1

        return assignments
