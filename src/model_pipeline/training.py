import torch
import torch.nn as nn
import math
import numpy as np

# Need to import these to patch on the fly during evaluation
from data_pipeline.release_preprocess import extract_patches_auto, transform_image


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


@torch.no_grad()
def evaluate_full_image(
    model, 
    device, 
    full_image, 
    full_label, 
    patch_size, 
    target_overlap, 
    loss_fn, 
    transform_name="none",
    label_min=None, 
    label_max=None,
    batch_size=32
):
    """
    Evaluates the model on a single FULL image by patching it, 
    averaging the patch predictions, and comparing to the full label.

    Args:
        model: Trained PyTorch model.
        device: 'cuda' or 'cpu'.
        full_image (np.ndarray): Full preprocessed image (H, W), normalized 0-1.
        full_label (float): The ground truth label for this full image (normalized 0-1).
        patch_size (int): Size of patches.
        target_overlap (float): Overlap percentage.
        loss_fn: Loss function.
        transform_name: Transform to apply to patches.
        label_min, label_max: For denormalization (optional).
        batch_size: Batch size for internal patch inference.

    Returns:
        dict: Metrics for this specific image, or None if image too small.
    """
    model.eval()

    # 1. Patch the full image dynamically
    patches = extract_patches_auto(full_image, patch_size=patch_size, target_overlap=target_overlap)
    
    if not patches:
        return None 

    # 2. Apply Transform if needed (same as training)
    if transform_name != "none":
        patches = [transform_image(p, transform_name) for p in patches]

    # 3. Prepare Patches for Model
    patches_np = np.array(patches) # (N, H, W)
    patches_tensor = torch.from_numpy(patches_np).float()
    
    # Add channel dimension: (N, H, W) -> (N, 1, H, W)
    if patches_tensor.ndim == 3:
        patches_tensor = patches_tensor.unsqueeze(1)
        
    # 4. Batched Inference
    all_preds = []
    num_patches = patches_tensor.size(0)
    
    for i in range(0, num_patches, batch_size):
        batch = patches_tensor[i : i + batch_size].to(device)
        preds = model(batch)
        all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds)
    
    # 5. Aggregate: Average the scores
    avg_pred = all_preds.mean().item()

    # 6. Calculate Metrics
    # Loss (Normalized)
    pred_t = torch.tensor([avg_pred], device=device)
    label_t = torch.tensor([full_label], device=device)
    loss = loss_fn(pred_t, label_t).item()

    # Denormalize
    if label_min is not None and label_max is not None:
        pred_microns = avg_pred * (label_max - label_min) + label_min
        target_microns = full_label * (label_max - label_min) + label_min
    else:
        pred_microns = avg_pred
        target_microns = full_label

    # Absolute Error & MAPE
    abs_error = abs(pred_microns - target_microns)
    mape = (abs_error / (target_microns + 1e-7)) * 100

    return {
        "loss": loss,
        "pred_microns": pred_microns,
        "target_microns": target_microns,
        "abs_error_microns": abs_error,
        "mape_pct": mape
    }


def evaluate_dataset_full_images(model, full_pairs, device, loss_fn, cfg, label_min, label_max):
    """
    Runs full image evaluation over a list of (image, label) pairs.
    """
    results = []
    
    for full_img, full_lbl in full_pairs:
        res = evaluate_full_image(
            model, 
            device, 
            full_img, 
            full_lbl, 
            patch_size=cfg.patch_size, 
            target_overlap=cfg.target_overlap, 
            loss_fn=loss_fn,
            transform_name=getattr(cfg, "transform_name", "none"),
            label_min=label_min,
            label_max=label_max
        )
        if res:
            results.append(res)
            
    if not results:
        return {}

    # Average metrics across the dataset
    avg_loss = np.mean([r['loss'] for r in results])
    avg_mae = np.mean([r['abs_error_microns'] for r in results])
    avg_mape = np.mean([r['mape_pct'] for r in results])
    
    return {
        "full_loss": avg_loss,
        "full_mae_microns": avg_mae,
        "full_mape_pct": avg_mape
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
