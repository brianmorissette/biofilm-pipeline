# src/sweep_runner.py
import wandb
import torch
from pathlib import Path

from data_pipeline.build_dataset import get_dataloaders
from model_pipeline.models import SurfaceAreaCNN
from model_pipeline.training import train_one_epoch, evaluate_v2, get_loss_fn, evaluate_dataset_full_images


def run(config):
    """
    Runs a single sweep experiment.

    Args:
        config: WandB configuration object containing hyperparameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model = SurfaceAreaCNN(
        image_size=config.patch_size,
        first_layer_channels=config.first_layer_channels,
        dropout=config.dropout,
        weight_decay=config.weight_decay,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = get_loss_fn(config.loss_fn)

    # Get dataloaders
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "raw_data_reorganized"

    train_loader, validation_loader, test_loader, val_label_min, val_label_max, val_full_pairs, test_full_pairs = (
        get_dataloaders(root=str(data_root), cfg=config)
    )

    print(
        f"Train batches: {len(train_loader)}, Validation batches: {len(validation_loader)}"
    )
    print(
        f"Train samples: {len(train_loader.dataset)}, Validation samples: {len(validation_loader.dataset)}, Test samples: {len(test_loader.dataset)}"
    )

    # Train and evaluate
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn)

        metrics = evaluate_v2(
            model,
            validation_loader,
            device,
            loss_fn,
            label_min=val_label_min,
            label_max=val_label_max,
        )
        val_loss = metrics["loss"]
        val_mae_microns = metrics["mae_microns"]
        val_rmse_microns = metrics["rmse_microns"]
        val_mape_pct = metrics["mape_pct"]

        # Full Image Evaluation
        full_image_metrics = evaluate_dataset_full_images(
            model,
            val_full_pairs,
            device,
            loss_fn,
            config,
            label_min=val_label_min,
            label_max=val_label_max,
        )
        val_full_loss = full_image_metrics["full_loss"]
        val_full_mae = full_image_metrics["full_mae_microns"]
        val_full_mape = full_image_metrics["full_mape_pct"]

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "validation/loss": val_loss,
                "validation/mae": val_mae_microns,
                "validation/rmse": val_rmse_microns,
                "validation/mape": val_mape_pct,
                "validation/full_loss": val_full_loss,
                "validation/full_mae": val_full_mae,
                "validation/full_mape": val_full_mape,
            }
        )

        print(
            f"Epoch {epoch:02d} | train/loss: {train_loss:.5f} | val/loss: {val_loss:.5f} | val/mae: {val_mae_microns:.5f} | val/mape: {val_mape_pct:.5f} | val/full_loss: {val_full_loss:.5f} | val/full_mae: {val_full_mae:.5f} | val/full_mape: {val_full_mape:.5f}"
        )


def main():
    """Initializes WandB and starts the sweep run."""
    wandb.init(project="biofilm-surface-area-cnn-sweep-v2")
    run(wandb.config)


if __name__ == "__main__":
    main()
