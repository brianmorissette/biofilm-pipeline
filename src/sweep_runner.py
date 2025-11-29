# src/sweep_runner.py
import wandb
import torch
import torch.nn as nn

from data_pipeline.build_dataset import get_dataloaders
from model_pipeline.cnn_models import SurfaceAreaCNN
from model_pipeline.training import train_one_epoch, evaluate, get_loss_fn

def run(config):
    # print the config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set model and optimizer
    model = SurfaceAreaCNN(image_size=config.patch_size, first_layer_channels=config.first_layer_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Set loss function
    loss_fn = get_loss_fn(config.loss_fn)

    # Get dataloaders
    train_loader, validation_loader = get_dataloaders(root="raw_data_reorganized", cfg=config)

    # Print dataloader sizes
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(validation_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}, Validation samples: {len(validation_loader.dataset)}")
    
    # Initialize best metrics
    best_test_rmse = float("inf")
    best_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        # train_one_epoch returns average loss over the dataset (MSE or L1 depending on loss_fn)
        train_loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        
        # evaluate returns (loss, mae, rmse)
        # loss corresponds to loss_fn
        val_loss, val_mae, val_rmse = evaluate(model, validation_loader, device, loss_fn)
        
        # Only log the requested metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,          # "Is the model actually learning?"
            "validation/loss": val_loss,       # "Is the model actually learning?"
            "validation/rmse": val_rmse,       # "Human Score"
            "validation/mae": val_mae,         # "Helpful for outliers"
        })
        
        print(f"Epoch {epoch:02d} | train/loss: {train_loss:.5f} | val/loss: {val_loss:.5f} | val/rmse: {val_rmse:.5f} | val/mae: {val_mae:.5f}")

        if val_rmse < best_test_rmse:
            best_test_rmse = val_rmse
            best_epoch = epoch
            print(f"  ✓ New best RMSE: {best_test_rmse:.5f} at epoch {best_epoch}")

    # Log final sample predictions to console for inspection (optional, but useful for user)
    with torch.no_grad():
        xq, yq = next(iter(validation_loader))
        xq = xq.to(device)
        pq = model(xq).cpu()
        
        print("\nSample predictions vs targets (first 8):")
        for i in range(min(8, len(pq))):
            pred = pq[i].item()
            target = yq[i].item()
            error = abs(pred - target)
            rel_error = (error / target * 100) if target != 0 else 0
            print(f"  [{i+1}] pred={pred:.4f}  target={target:.4f}  error={error:.4f}  ({rel_error:.1f}%)")
        
    
    # Optional: Log best metric to summary for table sorting
    wandb.summary["best_validation_rmse"] = best_test_rmse


def main():
    wandb.init(project="biofilm-surface-area")
    run(wandb.config)


if __name__ == "__main__":
    main()
