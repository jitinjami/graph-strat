import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops


class _H2GCNProp(MessagePassing):
    """Symmetric D^{-1/2} A D^{-1/2} aggregation with precomputed edge weights."""

    def __init__(self) -> None:
        super().__init__(aggr="add")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j


class H2GCN(torch.nn.Module):
    """H2GCN: Ego-neighbor separation + faithful K-hop aggregation + concatenation.

    Reference: Zhu et al., "Beyond Homophily in Graph Neural Networks:
    Current Limitations and Effective Designs", NeurIPS 2020.
    https://arxiv.org/abs/2006.11468

    K acts as depth: for each k in 1..K, strict-k-hop neighbors (nodes
    reachable in exactly k steps, not fewer) of the *original* ego embedding
    r^(0) are aggregated independently. The final representation is:

        cat(r^(0), r^(1), ..., r^(K))  →  classifier

    Adjacencies are stored as (edge_index, edge_weight) pairs so they move
    cleanly to any device (CPU / CUDA / MPS) via standard tensor ops.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        num_layers: int = 1,
        K: int = 2,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        dims = [in_channels] + [hidden_channels] * num_layers
        self.lins = ModuleList(Linear(dims[i], dims[i + 1]) for i in range(num_layers))
        self.lin_out = Linear((K + 1) * hidden_channels, out_channels)
        self.bn = BatchNorm1d(hidden_channels) if use_bn else None
        self.prop = _H2GCNProp()
        self.dropout = dropout
        self.K = K
        # Lazily populated on first forward: list of (edge_index, edge_weight) or None
        self._hop_edges: list | None = None

    # ------------------------------------------------------------------
    # Adjacency precomputation
    # ------------------------------------------------------------------

    def _init_adj(self, edge_index: torch.Tensor, n: int) -> None:
        """Build strict-k-hop GCN-normalised adjacencies for k = 1 … K.

        Stored as CPU (edge_index, edge_weight) tuples; moved to the right
        device inside forward() via normal tensor .to(device) calls.
        """
        ei, _ = remove_self_loops(edge_index)
        src_np = ei[0].cpu().numpy()
        dst_np = ei[1].cpu().numpy()
        vals = np.ones(len(src_np), dtype=np.float32)

        # Binary symmetric adjacency, no self-loops
        adj_sp = scipy.sparse.csr_matrix((vals, (src_np, dst_np)), shape=(n, n))
        adj_sp = ((adj_sp + adj_sp.T) > 0).astype(np.float32)
        adj_sp.setdiag(0)
        adj_sp.eliminate_zeros()

        self._hop_edges = []
        adj_pow = adj_sp.copy()     # tracks A^k
        cumulative = adj_sp.copy()  # union of hops 1..(k-1)

        for k in range(1, self.K + 1):
            if k == 1:
                strict = adj_sp.copy()
            else:
                adj_pow = adj_pow @ adj_sp
                strict = (adj_pow - cumulative)
                strict = (strict > 0).astype(np.float32)
                strict.setdiag(0)
                strict.eliminate_zeros()
                cumulative = ((cumulative + adj_pow) > 0).astype(np.float32)

            if strict.nnz == 0:
                self._hop_edges.append(None)
                continue

            coo = strict.tocoo()
            row = torch.from_numpy(coo.row.astype(np.int64))
            col = torch.from_numpy(coo.col.astype(np.int64))
            ei_k = torch.stack([row, col], dim=0)

            # GCN normalisation: D^{-1/2} A D^{-1/2}
            deg = torch.zeros(n).scatter_add_(0, col, torch.ones(col.size(0)))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt.isinf()] = 0.0
            ew_k = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            self._hop_edges.append((ei_k, ew_k))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embed → aggregate each strict hop from r^(0) → concat → classify."""
        if self._hop_edges is None:
            self._init_adj(edge_index, x.size(0))

        device = x.device

        # Ego embedding r^(0) via MLP
        for lin in self.lins:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(lin(x))
        if self.bn is not None:
            x = self.bn(x)
        r0 = x

        # r^(k) = strict-k-hop aggregation of r^(0), all hops independent
        reps = [r0]
        for hop in self._hop_edges:
            if hop is not None:
                ei_k, ew_k = hop
                reps.append(self.prop(r0, ei_k.to(device), ew_k.to(device)))
            else:
                reps.append(torch.zeros_like(r0))

        out = torch.cat(reps, dim=-1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin_out(out)
