import torch
import torch.nn as nn
import math


def train_one_epoch(model, loader, device, optimizer, loss_fn):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The data loader.
        device: The device to train on.
        optimizer: The optimizer.
        loss_fn: The loss function.

    Returns:
        The average loss for the epoch.
    """
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
    """
    Evaluates the model.

    Args:
        model: The PyTorch model.
        loader: The data loader.
        device: The device to evaluate on.
        loss_fn: The loss function.

    Returns:
        Tuple of (loss, mae, rmse).
    """
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


@torch.no_grad()
def evaluate_v2(model, loader, device, loss_fn, label_min=None, label_max=None):
    """
    Evaluates the model with more metrics and optional denormalization.

    Args:
        model: The PyTorch model.
        loader: The data loader.
        device: The device to evaluate on.
        loss_fn: The loss function.
        label_min: Minimum label value for denormalization (optional).
        label_max: Maximum label value for denormalization (optional).

    Returns:
        Dictionary containing 'loss', 'mae_microns', 'rmse_microns', 'mape_pct'.
    """
    model.eval()
    total_loss = 0.0

    # We will store all denormalized predictions and actuals to calculate MAPE and R2
    all_preds_microns = []
    all_targets_microns = []

    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # 1. Calculate Loss on NORMALIZED data (for the optimizer/math)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)

        # 2. Denormalize to Microns (for human interpretation)
        # If no min/max provided, we stay in normalized space (0-1)
        if label_min is not None and label_max is not None:
            pred_microns = pred * (label_max - label_min) + label_min
            target_microns = y * (label_max - label_min) + label_min
        else:
            pred_microns, target_microns = pred, y

        all_preds_microns.append(pred_microns.cpu())
        all_targets_microns.append(target_microns.cpu())
        n += x.size(0)

    # Concatenate all batches
    all_preds_microns = torch.cat(all_preds_microns)
    all_targets_microns = torch.cat(all_targets_microns)

    # 3. Calculate Interpretable Metrics
    abs_error = torch.abs(all_preds_microns - all_targets_microns)

    mae_microns = abs_error.mean().item()
    rmse_microns = torch.sqrt(torch.square(abs_error).mean()).item()

    # MAPE: Mean Absolute Percentage Error (The "off by X%" metric)
    # Added 1e-7 to avoid division by zero if a target is 0
    mape = (abs_error / (all_targets_microns + 1e-7)).mean().item() * 100

    return {
        "loss": total_loss / n,  # Normalized (for math)
        "mae_microns": mae_microns,  # Actual Microns
        "rmse_microns": rmse_microns,  # Actual Microns
        "mape_pct": mape,  # Percentage
    }




def get_loss_fn(loss_fn_name):
    """
    Factory function for loss functions.

    Args:
        loss_fn_name: Name of the loss function ("MSELoss", "L1Loss", "HuberLoss").

    Returns:
        The PyTorch loss function.

    Raises:
        ValueError: If the loss function name is invalid.
    """
    if loss_fn_name == "MSELoss":
        return nn.MSELoss()
    elif loss_fn_name == "L1Loss":
        return nn.L1Loss()
    else:
        raise ValueError(f"Invalid loss function: {loss_fn_name}")
