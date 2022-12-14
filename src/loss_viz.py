import torch
from typing import Collection, Union
import numpy as np
from matplotlib import pyplot as plt

'''
Visualizing the loss landscape of a neural network
Implementation from the paper
"Visualizing the loss landscape of neural nets"
https://arxiv.org/abs/1712.09913
As implemented by Alex Olar
https://olaralex.com/visualizing-the-loss-landscape/
'''

def init_model(
    model:torch.nn.Module,
    deltas:Collection[torch.Tensor, torch.Tensor],
    alpha:float,
    beta:float,
    device:Union[torch.device, str]
    ) -> None:
    with torch.no_grad():
        for param, delta in zip(model.parameters(), deltas):
            param.add_(delta[0], alpha=alpha).add_(delta[1], alpha=beta)
            param.add_(beta, torch.randn_like(param, device=device))


def init_directions(
    model:torch.nn.Module,
    device:Union[torch.device, str]
) -> Collection[torch.Tensor]:
    model.to(device)
    deltas = []
    for param in model.parameters():
        delta = torch.randn([2] + list(param.shape), device=device)
        # calculate norms and pack them into a tensor of the same shape as delta
        delta_norms = delta.view(2, -1).norm(dim=1, keepdim=True)
        delta_norms = delta_norms.view(2, *([1] * (len(param.shape)))).expand_as(delta)
        # normalize delta as indicated by the paper
        delta.div_(delta_norms).mul_(param.norm())
        deltas.append(delta)
    return deltas

def compute_loss_landscape(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss:torch.nn.Module,
    meshgrid_x:torch.Tensor,
    meshgrid_y:torch.Tensor,
    resolution:int,
    device:Union[torch.device, str]
) -> torch.Tensor:
    loss_values = torch.empty_like(meshgrid_x)
    deltas = init_directions(model, device)
    for i in range(resolution):
        for j in range(resolution):
            loss_total = 0.0
            alpha = meshgrid_x[i, j]
            beta = meshgrid_y[i, j]
            init_model(model, deltas, alpha, beta, device)
            num_datapoints = 0
            for x, y in dataloader:
                with torch.no_grad():
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    loss_value = loss(y_hat, y)
                    loss_total += loss_value.item()
                    num_datapoints += len(x)
            loss_values[i, j] = loss_total / num_datapoints
    return loss_values

def plot_viz(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss:torch.nn.Module,
    save_path:str=None,
    resolution:int=50,
    device:Union[torch.device, str]="cpu"
):
    meshgrid_x = torch.linspace(-1, 1, resolution)
    meshgrid_y = torch.linspace(-1, 1, resolution)
    meshgrid_x, meshgrid_y = torch.meshgrid(meshgrid_x, meshgrid_y)
    loss_values = compute_loss_landscape(model, dataloader, loss, meshgrid_x, meshgrid_y, resolution, device)
    plt.figure(figsize=(10, 10))
    plt.title("Loss landscape")
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.contourf(meshgrid_x, meshgrid_y, loss_values, 100)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


