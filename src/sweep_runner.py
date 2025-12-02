# src/sweep_runner.py
import wandb
import torch
import torch.nn as nn
import os
from pathlib import Path

from data_pipeline.build_dataset import get_dataloaders
from model_pipeline.cnn_models import SurfaceAreaCNN
from model_pipeline.training import train_one_epoch, evaluate, get_loss_fn
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
        
    train_loader, validation_loader = get_dataloaders(root=str(data_root), cfg=config)

    # print dataloader sizes
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(validation_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}, Validation samples: {len(validation_loader.dataset)}")
    
    # initialize best metrics
    best_test_rmse = float("inf")
    best_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        # train one epoch
        train_loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        
        # evaluate on validation set
        val_loss, val_mae, val_rmse = evaluate(model, validation_loader, device, loss_fn)
        
        # log the metrics to WandB
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

    # log final sample predictions to console for inspection
    with torch.no_grad():
        sample_inputs, sample_targets = next(iter(validation_loader))
        sample_inputs = sample_inputs.to(device)
        sample_predictions = model(sample_inputs).cpu()
        
        print("\nSample predictions vs targets (first 8):")
        for idx in range(min(8, len(sample_predictions))):
            prediction = sample_predictions[idx].item()
            target = sample_targets[idx].item()
            abs_error = abs(prediction - target)
            rel_error_pct = (abs_error / target * 100) if target != 0 else 0
            print(f"  [{idx+1}] pred={prediction:.4f}  target={target:.4f}  error={abs_error:.4f}  ({rel_error_pct:.1f}%)")
        
    # # ---------------------------------------------------------
    # # Final Evaluation: Image-Level Predictions (Aggregated)
    # # ---------------------------------------------------------
    # print("\n" + "="*60)
    # print("FINAL EVALUATION: Aggregated Image-Level Predictions")
    # print("="*60)

    # # Re-load raw data manually to get full images (since get_dataloaders doesn't return them)
    # # We rely on random_state=42 to ensure the split matches training
    # biofilm_images = load_images(data_root / "biofilm")
    # release_images = load_images(data_root / "release")
    # raw_pairs = list(zip(biofilm_images, release_images))
    
    # # Reproduce the split logic
    # train_raw, test_raw = train_test_split(raw_pairs, train_size=0.9, random_state=42)
    # _, validation_raw = train_test_split(train_raw, train_size=0.8, random_state=42)
    
    # # We also need the label normalization stats from the full dataset (or at least training set)
    # # to correctly interpret the target labels. For simplicity here, we re-calculate
    # # stats on the FULL set of labels, which is what build_dataset does by default.
    # from data_pipeline.biofilm_preprocess import preprocess_biofilm, threshold_image, get_biofilm_label
    
    # # Calculate label stats to normalize targets correctly
    # all_labels = []
    # # Note: In a perfect world we only use train_raw for stats, but build_dataset currently 
    # # calculates stats on whatever list is passed to _build_pairs. 
    # # To perfectly match validation_loader's labels, we need to know exactly how they were normalized.
    # # Assuming build_dataset normalizes based on the subset passed to it:
    # subset_labels = []
    # for b, _ in validation_raw:
    #     p_b = preprocess_biofilm(b)
    #     thresh = threshold_image(p_b, config.threshold_method)
    #     lbl = get_biofilm_label(p_b, thresh, "surface area")
    #     subset_labels.append(lbl)
    
    # subset_labels = np.array(subset_labels, dtype=np.float32)
    # min_val, max_val = subset_labels.min(), subset_labels.max()

    # model.eval()
    # image_level_sq_errors = []
    # image_level_abs_errors = []

    # print(f"{'Image ID':<10} | {'Pred (Avg)':<12} | {'Target':<10} | {'Error':<10} | {'Patches':<8}")
    # print("-" * 60)

    # for i, (biofilm_img, release_img) in enumerate(validation_raw):
    #     # 1. Prepare Target Label
    #     raw_target = subset_labels[i]
    #     if max_val - min_val == 0:
    #         target_label = 0.0
    #     else:
    #         target_label = (raw_target - min_val) / (max_val - min_val)

    #     # 2. Prepare Input Image
    #     gray_release = grayscale(release_img)
    #     norm_release = normalize(gray_release)

    #     # 3. Extract Patches
    #     if config.patch_method == "robust":
    #         patches = extract_patches_robust(norm_release, config.patch_size, config.stride_multiplier)
    #     else:
    #         patches = extract_patches(norm_release, config.patch_size, config.stride_multiplier)

    #     # 4. Apply Transforms
    #     if config.transform != "none":
    #         patches = np.array([get_transform(p, config.transform) for p in patches])

    #     if len(patches) == 0:
    #         continue

    #     # 5. Predict
    #     batch_x = torch.from_numpy(patches.astype(np.float32)).unsqueeze(1).to(device)
        
    #     with torch.no_grad():
    #         preds = model(batch_x)
        
    #     # 6. Aggregate
    #     avg_pred = preds.mean().item()
        
    #     # 7. Calculate Error
    #     abs_err = abs(avg_pred - target_label)
    #     sq_err = (avg_pred - target_label) ** 2
        
    #     image_level_abs_errors.append(abs_err)
    #     image_level_sq_errors.append(sq_err)

    #     if i < 10:
    #         print(f"Img {i+1:<6} | {avg_pred:.4f}       | {target_label:.4f}     | {abs_err:.4f}     | {len(patches):<8}")

    # # Final Metrics
    # if len(image_level_sq_errors) > 0:
    #     final_rmse = sum(image_level_sq_errors) / len(image_level_sq_errors) ** 0.5 # Typo fix: sqrt of MSE
    #     final_rmse = (sum(image_level_sq_errors) / len(image_level_sq_errors)) ** 0.5
    #     final_mae = sum(image_level_abs_errors) / len(image_level_abs_errors)
        
    #     print("-" * 60)
    #     print(f"Aggregated Validation RMSE: {final_rmse:.5f}")
    #     print(f"Aggregated Validation MAE:  {final_mae:.5f}")
        
    #     wandb.log({
    #         "final/aggregated_rmse": final_rmse,
    #         "final/aggregated_mae": final_mae
    #     })
    #     wandb.summary["aggregated_rmse"] = final_rmse
    # # ---------------------------------------------------------
    # # Final Evaluation: Image-Level Predictions (Aggregated)
    # # ---------------------------------------------------------
    

    # Optional: Log best metric to summary for table sorting
    wandb.summary["best_validation_rmse"] = best_test_rmse


def main():
    wandb.init(project="biofilm-surface-area-cnn-sweep")
    run(wandb.config)


if __name__ == "__main__":
    main()
