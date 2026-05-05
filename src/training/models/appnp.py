import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import APPNP


class APPNPModel(torch.nn.Module):
    """APPNP: Predict then Propagate with configurable MLP depth (Gasteiger et al., ICLR 2019).

    An num_layers-deep MLP transforms features, then personalised PageRank propagation
    (K steps, teleport probability alpha) diffuses the predictions over the graph.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        K: int = 10,
        alpha: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.lins = ModuleList(Linear(dims[i], dims[i + 1]) for i in range(num_layers))
        self.prop = APPNP(K=K, alpha=alpha)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """MLP transform (dropout → Linear → ReLU per hidden layer) then APPNP propagation. Returns raw logits."""
        for lin in self.lins[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = self.prop(x, edge_index)
        return x
