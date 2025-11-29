# train_surface_area_cnn.py
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add src to path so imports work
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from final_data_pipeline import get_dataloaders
from model_pipeline.cnn_models import SurfaceAreaCNN
from model_pipeline.training import train_one_epoch, evaluate

cfg = {
    "batch_size": 32,
    "patch_size": 128,
    "stride_multiplier": 1,
    "threshold_method": "iterative",
    "shuffle_train": True,
    "epochs": 10,
    "learning_rate": 1e-3,
    "first_layer_channels": 8,
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SurfaceAreaCNN(image_size=cfg["patch_size"], first_layer_channels=cfg["first_layer_channels"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate=cfg["learning_rate"])
loss_fn = nn.MSELoss()

train_loader, test_loader = get_dataloaders(root="raw_data_reorganized", cfg=cfg)

# Quick shape sanity-check
xb, yb = next(iter(train_loader))
assert xb.shape[1:] == (1, cfg["patch_size"], cfg["patch_size"]), f"Expected (1,{cfg['patch_size']},{cfg['patch_size']}), got {xb.shape[1:]}"
print(f"Device: {device}, Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

best_test_rmse = float("inf")
for epoch in range(1, cfg["epochs"] + 1):
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

