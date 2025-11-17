# src/sweep_runner.py
import wandb
import torch
import torch.nn as nn
from final_data_pipeline import get_dataloaders
from model_pipeline.cnn_models import SurfaceAreaCNN
from model_pipeline.training import train_one_epoch, evaluate

def run(config):


    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurfaceAreaCNN(image_size=config.patch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    train_loader, test_loader = get_dataloaders(root="raw_data_reorganized", cfg=config)

    # # Quick shape sanity-check
    # xb, yb = next(iter(train_loader))
    # assert xb.shape[1:] == (1, cfg["patch_size"], cfg["patch_size"]), f"Expected (1,{cfg['patch_size']},{cfg['patch_size']}), got {xb.shape[1:]}"
    # print(f"Device: {device}, Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    best_test_rmse = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        test_mse, test_mae, test_rmse = evaluate(model, test_loader, device, loss_fn)
        print(f"Epoch {epoch:02d} | train MSE: {train_mse:.5f} | test MSE: {test_mse:.5f} | test MAE: {test_mae:.5f} | test RMSE: {test_rmse:.5f}")

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse

    # Show a small qualitative sample
    with torch.no_grad():
        xq, yq = next(iter(test_loader))
        xq = xq.to(device)
        pq = model(xq).cpu()
        print("Sample preds vs targets (first 8):")
        for i in range(min(8, len(pq))):
            print(f"  pred={pq[i].item():.4f}   target={yq[i].item():.4f}")

    # ---- LOG ----
    wandb.log({
        "test_rmse": best_test_rmse,
    })


    # # ---- MODEL ----
    # model = build_model(
    #     architecture=config.model_architecture,
    #     in_channels=config.in_channels,
    #     num_classes=config.num_classes,
    #     patch_size=config.patch_size
    # )

    # # ---- TRAIN ----
    # trainer = Trainer(
    #     model=model,
    #     lr=config.learning_rate,
    #     epochs=config.epochs
    # )

    # val_loss = trainer.fit(train_loader, val_loader)

    


def main():
    wandb.init(project="biofilm-surface-area")
    run(wandb.config)


if __name__ == "__main__":
    main()
