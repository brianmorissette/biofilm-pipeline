# src/sweep_runner.py
import wandb
import torch
import torch.nn as nn
import os
from pathlib import Path

from data_pipeline.build_dataset import get_dataloaders
from model_pipeline.cnn_models import SurfaceAreaCNN
from model_pipeline.training import train_one_epoch, evaluate, evaluate_v2, get_loss_fn
from utils import *
from sklearn.model_selection import train_test_split

def run(config):
    # print the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # initialize model
    model = SurfaceAreaCNN(
        image_size=config.patch_size, 
        first_layer_channels=config.first_layer_channels, 
        dropout=config.dropout, 
        weight_decay=config.weight_decay).to(device)
    
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # set loss function
    loss_fn = get_loss_fn(config.loss_fn)

    # get dataloaders
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "raw_data_reorganized"
        
    train_loader, validation_loader, test_loader, val_label_min, val_label_max = get_dataloaders(root=str(data_root), cfg=config)

    # print dataloader sizes
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(validation_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}, Validation samples: {len(validation_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # train and evaluatethe model
    for epoch in range(1, config.epochs + 1):
        # train one epoch
        train_loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        
        # evaluate on validation set
        metrics = evaluate_v2(model, validation_loader, device, loss_fn, label_min=val_label_min, label_max=val_label_max)
        val_loss = metrics["loss"]
        val_mae_microns = metrics["mae_microns"]
        val_rmse_microns = metrics["rmse_microns"]
        val_mape_pct = metrics["mape_pct"]

        # log the metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,          # "Is the model actually learning?"
            "validation/loss": val_loss,       # "Is the model actually learning?"
            "validation/mae": val_mae_microns,       # "Human Score"
            "validation/rmse": val_rmse_microns,         # "Helpful for outliers"
            "validation/mape": val_mape_pct,              # "Percentage error"
        })
        
        print(f"Epoch {epoch:02d} | train/loss: {train_loss:.5f} | val/loss: {val_loss:.5f} | val/mae: {val_mae_microns:.5f} | val/rmse: {val_rmse_microns:.5f} | val/mape: {val_mape_pct:.5f}")


def main():
    wandb.init(project="biofilm-surface-area-cnn-sweep-v2")
    run(wandb.config)


if __name__ == "__main__":
    main()
