# src/sweep_runner.py
import wandb
import torch
import torch.nn as nn

from final_data_pipeline import get_dataloaders
from model_pipeline.cnn_models import SurfaceAreaCNN
from model_pipeline.training import train_one_epoch, evaluate

def run(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = SurfaceAreaCNN(image_size=config.patch_size, first_layer_channels=config.first_layer_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (trainable: {num_trainable_params:,})")

    train_loader, validation_loader = get_dataloaders(root="raw_data_reorganized", cfg=config)
    
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(validation_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}, Validation samples: {len(validation_loader.dataset)}")
    
    # Log setup metrics
    wandb.log({
        "setup/device": str(device),
        "setup/total_parameters": num_params,
        "setup/trainable_parameters": num_trainable_params,
        "setup/train_batches": len(train_loader),
        "setup/validation_batches": len(validation_loader),
        "setup/train_samples": len(train_loader.dataset),
        "setup/validation_samples": len(validation_loader.dataset),
    })

    best_test_rmse = float("inf")
    best_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        test_mse, test_mae, test_rmse = evaluate(model, validation_loader, device, loss_fn)
        
        # Log per-epoch metrics with more detail
        train_rmse = train_mse ** 0.5
        wandb.log({
            "epoch": epoch,
            "train/mse": train_mse,
            "train/rmse": train_rmse,
            "validation/mse": test_mse,
            "validation/mae": test_mae,
            "validation/rmse": test_rmse,
            "best/validation_rmse": best_test_rmse,
        })
        
        print(f"Epoch {epoch:02d} | train MSE: {train_mse:.5f} (RMSE: {train_rmse:.5f}) | validation MSE: {test_mse:.5f} | validation MAE: {test_mae:.5f} | validation RMSE: {test_rmse:.5f}")

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_epoch = epoch
            # Log improvement
            wandb.log({
                "best/validation_rmse": best_test_rmse,
                "best/validation_mse": test_mse,
                "best/validation_mae": test_mae,
                "best/epoch": best_epoch,
            })
            print(f"  ✓ New best RMSE: {best_test_rmse:.5f} at epoch {best_epoch}")

    # Show a small qualitative sample and log predictions
    with torch.no_grad():
        xq, yq = next(iter(validation_loader))
        xq = xq.to(device)
        pq = model(xq).cpu()
        
        print("\nSample predictions vs targets (first 8):")
        predictions_list = []
        targets_list = []
        errors_list = []
        
        for i in range(min(8, len(pq))):
            pred = pq[i].item()
            target = yq[i].item()
            error = abs(pred - target)
            rel_error = (error / target * 100) if target != 0 else 0
            predictions_list.append(pred)
            targets_list.append(target)
            errors_list.append(error)
            print(f"  [{i+1}] pred={pred:.4f}  target={target:.4f}  error={error:.4f}  ({rel_error:.1f}%)")
        
        # Log sample predictions
        avg_sample_error = sum(errors_list) / len(errors_list) if errors_list else 0
        wandb.log({
            "samples/predictions": predictions_list,
            "samples/targets": targets_list,
            "samples/abs_errors": errors_list,
            "samples/avg_abs_error": avg_sample_error,
        })

    # Log final summary metrics
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Best validation RMSE: {best_test_rmse:.5f}")
    print(f"Best epoch: {best_epoch}/{config.epochs}")
    print(f"{'='*50}\n")
    
    wandb.log({
        "final/best_validation_rmse": best_test_rmse,
        "final/best_epoch": best_epoch,
        "final/improvement_over_baseline": 0,  # Can be updated if you have a baseline
    })
    
    # Set summary metrics for easy comparison across runs
    wandb.summary["best_validation_rmse"] = best_test_rmse
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["final_train_rmse"] = train_rmse



def main():
    wandb.init(project="biofilm-surface-area")
    run(wandb.config)


if __name__ == "__main__":
    main()
