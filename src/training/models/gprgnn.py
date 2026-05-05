import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GPR_prop(MessagePassing):
    """Generalised PageRank propagation with learnable per-hop weights.

    Computes:  h = sum_{k=0}^{K} gamma_k * (A_hat)^k * x
    where A_hat is the symmetrically normalised adjacency with self-loops
    and gamma_0 … gamma_K are learned scalar weights initialised according
    to ``Init``.

    Reference: Chien et al., "Adaptive Universal Generalized PageRank GNN",
    ICLR 2021. https://arxiv.org/abs/2006.07988
    """

    def __init__(self, K: int, alpha: float, Init: str, Gamma=None, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K     = K
        self.alpha = alpha
        self.Init  = Init
        self.Gamma = Gamma

        assert Init in {"SGC", "PPR", "NPPR", "Random", "WS"}, \
            f"Unknown Init scheme '{Init}'. Choose from: SGC, PPR, NPPR, Random, WS."

        TEMP = self._init_weights()
        self.temp = Parameter(torch.tensor(TEMP))

    def _init_weights(self) -> np.ndarray:
        K, alpha = self.K, self.alpha
        if self.Init == "SGC":
            TEMP = np.zeros(K + 1)
            TEMP[int(alpha)] = 1.0
        elif self.Init == "PPR":
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif self.Init == "NPPR":
            TEMP = alpha ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif self.Init == "Random":
            bound = np.sqrt(3 / (K + 1))
            TEMP  = np.random.uniform(-bound, bound, K + 1)
            TEMP  = TEMP / np.sum(np.abs(TEMP))
        else:  # WS
            TEMP = self.Gamma
        return TEMP

    def reset_parameters(self):
        with torch.no_grad():
            self.temp.copy_(torch.tensor(self._init_weights()))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight=None,
    ) -> torch.Tensor:
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K}, temp={self.temp})"


class GPRGNN(torch.nn.Module):
    """GPR-GNN: MLP feature transform + generalised PageRank propagation.

    The propagation weights gamma_0 … gamma_K are learned jointly with the
    MLP, allowing the model to adapt to both homophilic and heterophilic
    graphs by tuning the effective filter shape.

    ``num_layers`` controls MLP depth (same convention as GCN/SAGE/etc.).
    ``K`` controls the number of GPR propagation steps (model-specific config).

    Reference: Chien et al., "Adaptive Universal Generalized PageRank GNN",
    ICLR 2021. https://arxiv.org/abs/2006.07988
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        num_layers: int = 2,
        K: int = 10,
        alpha: float = 0.1,
        Init: str = "PPR",
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.lins = torch.nn.ModuleList(
            Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.prop    = GPR_prop(K=K, alpha=alpha, Init=Init)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """MLP (dropout → Linear → ReLU per hidden layer) → GPR propagation. Returns raw logits."""
        for lin in self.lins[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(lin(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return self.prop(x, edge_index)
