import torch
from torch import nn
from torch.nn import functional as F

from .tgcncell import TGCNCell


class TGCN(nn.Module):
	"""
	Source: https://arxiv.org/pdf/1811.05320.
	"""
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, layers_nb: int = 2, output_activation: nn.Module = None, use_gat: bool = True):
		super(TGCN, self).__init__()
		self.hidden_size = hidden_size
		self.layers_nb = max(1, layers_nb)
		self.cells = nn.ModuleList(
			[TGCNCell(in_channels, hidden_size, use_gat=use_gat)] + [TGCNCell(hidden_size, hidden_size, use_gat=use_gat) for _ in range(self.layers_nb - 1)]
		)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
		h_prev = [
			torch.zeros(x.shape[0], self.hidden_size, device=x.device) for _ in range(self.layers_nb)
		]
		for t in range(x.shape[-1]):
			h = x[:, :, t]  # h is the output of the previous GRU layer (the input features for the first layer)
			for i, cell in enumerate(self.cells):
				h = cell(h, edge_index, edge_weight, h_prev[i])
				h_prev[i] = h
		return self.out(h_prev[-1])
