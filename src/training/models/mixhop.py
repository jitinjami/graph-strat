from typing import List

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList
from torch_geometric.nn import MixHopConv


class MixHop(torch.nn.Module):
    """MixHop with configurable depth (Abu-El-Haija et al., ICML 2019).

    Each MixHop layer aggregates adjacency powers {0, 1, 2} and is followed by
    BatchNorm and ReLU. A final linear layer maps to out_channels.

    hidden_channels is the per-power output width; total intermediate width per
    layer = len(powers) * (hidden_channels // len(powers)).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        powers: List[int] = None,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        if powers is None:
            powers = [0, 1, 2]
        p = hidden_channels // len(powers)
        total = len(powers) * p

        self.convs = ModuleList()
        self.bns = ModuleList()
        # First layer: in_channels → total
        self.convs.append(MixHopConv(in_channels, p, powers=powers))
        self.bns.append(BatchNorm1d(total))
        # Remaining layers: total → total
        for _ in range(num_layers - 1):
            self.convs.append(MixHopConv(total, p, powers=powers))
            self.bns.append(BatchNorm1d(total))

        self.lin = Linear(total, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """dropout → MixHop → BN → ReLU for each layer, then dropout → linear. Returns raw logits."""
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x
