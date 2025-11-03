import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader


def get_regression_error(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float, float, float]:
    mse = 0
    rmse = 0
    mae = 0
    mre = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_weight)
            mse += F.mse_loss(out, data.y).item()
            rmse += F.mse_loss(out, data.y).sqrt().item()
            mae += F.l1_loss(out, data.y).item()
            mre += (F.l1_loss(out, data.y) / data.y.abs().mean()).item()
    return mse / len(dataloader), rmse / len(dataloader), mae / len(dataloader), mre / len(dataloader)

def plot_regression(model: nn.Module, data: Data, device: torch.device, title: str = None) -> None:
    model.eval()
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title)
    
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
        
    stocks_idx = np.random.choice(data.x.shape[0] // (len(data.ptr) - 1), 4)

    preds = out.reshape(len(data.ptr) - 1, -1)
    target = data.y.reshape(len(data.ptr) - 1, -1)

    for idx, stock_idx in enumerate(stocks_idx):
        ax = axs[idx // 2, idx % 2]
        
        ax.plot(target[:, stock_idx].detach().cpu().numpy(), label="Real")
        ax.plot(preds[:, stock_idx].detach().cpu().numpy(), label="Predicted")
        
        ax.set_title(f"Stock {stock_idx}")
        ax.legend()

    plt.show()


def measure_accuracy(model: nn.Module, data: Data) -> float:
	out = model(data.x, data.edge_index, data.edge_weight)
	if out.shape[1] == 1:  # Binary classification
		return (F.sigmoid(out).round() == data.y).sum().item() / len(data.y)
	else:  # Multi-class classification
		return (F.softmax(out, dim=-1).argmax(dim=-1) == data.y).sum().item() / len(data.y)
