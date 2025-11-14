# src/sweep_runner.py
import wandb
from src.data_loading.dataset import get_dataloaders
from src.models.factory import build_model
from src.training.trainer import Trainer

def run(config):
    # ---- DATA ----
    train_loader, test_loader = get_dataloaders(
        root="data/raw",
        cfg=config
    )

    # ---- MODEL ----
    model = build_model(
        architecture=config.model_architecture,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        patch_size=config.patch_size
    )

    # ---- TRAIN ----
    trainer = Trainer(
        model=model,
        lr=config.learning_rate,
        epochs=config.epochs
    )

    val_loss = trainer.fit(train_loader, val_loader)

    # ---- LOG ----
    wandb.log({
        "val_loss": val_loss,
        "best_epoch": trainer.best_epoch,
    })


def main():
    wandb.init(project="biofilm-surface-area")
    run(wandb.config)


if __name__ == "__main__":
    main()
