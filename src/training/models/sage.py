import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """GraphSAGE with mean aggregation and configurable depth (Hamilton et al., NeurIPS 2017)."""

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
            SAGEConv(dims[i], dims[i + 1], aggr="mean") for i in range(num_layers)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """dropout → SAGEConv → ReLU for each hidden layer, then dropout → SAGEConv. Returns raw logits."""
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
