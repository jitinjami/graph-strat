import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """GCN with configurable depth (Kipf & Welling, ICLR 2017)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = torch.nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """dropout → GCNConv → ReLU for each hidden layer, then dropout → GCNConv. Returns raw logits."""
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
