import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN
from .gat import GAT


class TGCNCell(nn.Module):
	"""
	Source: https://arxiv.org/pdf/1811.05320.
	"""
	def __init__(self, in_channels: int, hidden_size: int, use_gat: bool = True):
		super(TGCNCell, self).__init__()
		if use_gat:
			self.gcn = GAT(in_channels, [hidden_size, hidden_size])
		else:
			self.gcn = GCN(in_channels, [hidden_size, hidden_size])
		self.lin_u = nn.Linear(2 * hidden_size + in_channels, hidden_size)
		self.lin_r = nn.Linear(2 * hidden_size + in_channels, hidden_size)
		self.lin_c = nn.Linear(2 * hidden_size + in_channels, hidden_size)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor, h: torch.tensor) -> tuple[
		torch.tensor, torch.tensor]:
		gcn_out = F.sigmoid(self.gcn(x, edge_index, edge_weight))  # f(A,X_t), Eq. 2
		u = F.sigmoid(self.lin_u(torch.cat([x, gcn_out, h], dim=-1)))  # u_t, Eq. 3
		r = F.sigmoid(self.lin_r(torch.cat([x, gcn_out, h], dim=-1)))  # r_t,  Eq. 4
		c = F.tanh(self.lin_c(torch.cat([x, gcn_out, r * h], dim=-1)))  # c_t, Eq. 5

		return u * h + (1 - u) * c  # h_t, Eq. 6
