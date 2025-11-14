# src/sweep_runner.py
import wandb
from src.data_loading.dataset import BiofilmDataset
from src.models.factory import build_model
from src.training.trainer import Trainer

def run(config):

    # --- DATA ---
    dataset = BiofilmDataset(
        patch_size=config.patch_size,
        threshold_method=config.threshold_method,
        # ...other preprocessing params
    )
    train_loader, val_loader = dataset.get_loaders(batch_size=config.batch_size)

    # --- MODEL ---
    model = build_model(
        architecture=config.model_architecture,
        depth=config.model_depth,
        filters=config.model_filters
    )

    # --- TRAIN ---
    trainer = Trainer(
        model=model,
        lr=config.learning_rate,
        epochs=config.epochs
    )

    trainer.fit(train_loader, val_loader)

    # --- Log ---
    wandb.log({
        "val_loss": trainer.val_loss,
        "val_mae": trainer.val_mae,
        "best_epoch": trainer.best_epoch
    })

def main():
    wandb.init(project="biofilm-surface-area", config=DEFAULT_CONFIG)
    run(wandb.config)

if __name__ == "__main__":
    main()
