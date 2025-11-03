import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
    
	def __init__(self, in_channels: int, layer_sizes: list[int] = None, bias: bool = True):
		super(GAT, self).__init__()
		layer_sizes = layer_sizes or [32, 32]
		self.convs = nn.ModuleList([
            GATv2Conv(in_channels, layer_sizes[0], bias=bias, edge_dim=1),
        ] + [
            GATv2Conv(layer_sizes[i], layer_sizes[i + 1], bias=bias, edge_dim=1) for i in
            range(len(layer_sizes) - 1)
        ])

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
		for conv in self.convs[:-1]:
			x = F.leaky_relu(conv(x, edge_index, edge_weight))
		return self.convs[-1](x, edge_index, edge_weight)
