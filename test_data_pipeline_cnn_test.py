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


# ---- Tiny CNN for 1x128x128 → scalar ----
class SurfaceAreaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 64x64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 16x16
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 1),                     # regression output
        )

    def forward(self, x):
        return self.head(self.feat(x)).squeeze(-1)  # [B]

def train_one_epoch(model, loader, device, optimizer, loss_fn):
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

def main():
    p = argparse.ArgumentParser(description="Train a simple CNN to regress surface area from release cell patches.")
    p.add_argument("--root", type=str, default="raw_data_reorganized", help="Root folder containing biofilm/ and release/")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--stride_multiplier", type=int, default=1)
    p.add_argument("--threshold_method", type=str, default="iterative")
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--save", type=str, default="surface_area_cnn.pt")
    args = p.parse_args()

    # Build loaders using your data_module (split first, then augment)
    cfg = {
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "stride_multiplier": args.stride_multiplier,
        "threshold_method": args.threshold_method,
        "shuffle_train": True,
        "epochs": args.epochs,
    }
    train_loader, test_loader = get_dataloaders(root=args.root, cfg=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurfaceAreaCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Quick shape sanity-check
    xb, yb = next(iter(train_loader))
    assert xb.shape[1:] == (1, args.patch_size, args.patch_size), f"Expected (1,{args.patch_size},{args.patch_size}), got {xb.shape[1:]}"
    print(f"Device: {device}, Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    best_test_rmse = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        train_mse = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        test_mse, test_mae, test_rmse = evaluate(model, test_loader, device, loss_fn)
        print(f"Epoch {epoch:02d} | train MSE: {train_mse:.5f} | test MSE: {test_mse:.5f} | test MAE: {test_mae:.5f} | test RMSE: {test_rmse:.5f}")

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            torch.save({"model_state": model.state_dict(),
                        "cfg": cfg}, args.save)
            print(f"  ✓ Saved best model → {args.save}")

    # Show a small qualitative sample
    with torch.no_grad():
        xq, yq = next(iter(test_loader))
        xq = xq.to(device)
        pq = model(xq).cpu()
        print("Sample preds vs targets (first 8):")
        for i in range(min(8, len(pq))):
            print(f"  pred={pq[i].item():.4f}   target={yq[i].item():.4f}")

if __name__ == "__main__":
    main()
