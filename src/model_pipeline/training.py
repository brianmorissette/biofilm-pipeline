import torch
import torch.nn as nn
import math

def train_one_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, total_mae, total_sqerr, n = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
        total_mae += torch.abs(pred - y).sum().item()
        total_sqerr += torch.square(pred - y).sum().item()
        n += x.size(0)
    mse = total_sqerr / n
    rmse = math.sqrt(mse)
    return total_loss / n, total_mae / n, rmse

def get_loss_fn(loss_fn_name):
    if loss_fn_name == "MSELoss":
        return nn.MSELoss()
    elif loss_fn_name == "L1Loss":
        return nn.L1Loss()
    elif loss_fn_name == "HuberLoss":
        return nn.HuberLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_fn_name}")
