import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """GAT with configurable depth (Velickovic et al., ICLR 2018).

    All layers except the last use `heads` attention heads with concat=True,
    keeping the total output width equal to hidden_channels.
    The final layer uses a single head with concat=False to produce out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        if hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
            )
        head_dim = hidden_channels // heads
        self.convs = torch.nn.ModuleList()
        # First layer: in_channels → hidden_channels (via heads × head_dim)
        self.convs.append(GATConv(in_channels, head_dim, heads=heads, dropout=dropout))
        # Middle layers: hidden_channels → hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, head_dim, heads=heads, dropout=dropout))
        # Last layer: hidden_channels → out_channels, single head
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """dropout → multi-head GATConv → ELU for each hidden layer, then dropout → single-head GATConv. Returns raw logits."""
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
