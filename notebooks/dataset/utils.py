import numpy as np
import pandas as pd
import torch


def get_graph_in_pyg_format(values_path: str, adj_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
	adj = np.load(adj_path)
	nodes_nb, edge_nb = len(adj), np.count_nonzero(adj)
	x = torch.tensor(
		values.drop(columns=["Close"]).to_numpy().reshape((nodes_nb, -1, values.shape[1] - 1)), dtype=torch.float32
	)
	x = x.transpose(1, 2)
	close_prices = torch.tensor(
		values[["Close"]].to_numpy().reshape((nodes_nb, -1)), dtype=torch.float32
	)
	edge_index, edge_weight = torch.zeros((2, edge_nb), dtype=torch.long), torch.zeros((edge_nb,), dtype=torch.float32)
	count = 0
	for i in range(nodes_nb):
		for j in range(nodes_nb):
			if (weight := adj[i, j]) != 0:
				edge_index[0, count], edge_index[1, count] = i, j
				edge_weight[count] = weight
				count += 1

	return x, close_prices, edge_index, edge_weight


def get_stocks_labels() -> list[str]:
	return pd.read_csv("../data/raw/values.csv")["Symbol"].unique().tolist()
