# data_module.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# local preprocessing utilities
from data_pipeline.release_preprocess import *
from data_pipeline.biofilm_preprocess import *
from utils import *


# -----------------------------
# Dataset: wraps (img, label) pairs for a CNN
# -----------------------------
class ImageLabelDataset(Dataset):
    def __init__(self, samples):
        # samples: list of (np.ndarray(H,W) float in [0,1], float)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, y = self.samples[i]
        # ensure single-channel, channel-first (1,H,W)
        if img.ndim == 2:
            img = img[None, ...]
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.transpose(img, (2, 0, 1))
        x = torch.from_numpy(img.astype(np.float32))   # image → float32 tensor
        y = torch.tensor(y, dtype=torch.float32)       # label → float32 tensor (regression)
        return x, y


# -----------------------------
# Build (img,label) pairs from raw images (biofilm, release)
# -----------------------------
def _build_pairs(raw_pairs, threshold_method, patch_size, patch_stride):
    # 1) per-image preprocessing + label (no patches yet)
    pre_patch_pairs = []
    for biofilm, release in raw_pairs:
        # release → grayscale + normalize
        grayscale_release = grayscale(release)
        normalized_release = normalize(grayscale_release)

        # biofilm → preprocess + threshold + label (surface area)
        preprocessed_biofilm = preprocess_biofilm(biofilm)
        threshold = threshold_image(preprocessed_biofilm, threshold_method=threshold_method)
        biofilm_label = get_biofilm_label(preprocessed_biofilm, threshold, label="surface area")

        pre_patch_pairs.append((normalized_release, biofilm_label))

    # 2) extract patches + rotations (original + 90/180/270)
    samples = []
    for release, biofilm_label in pre_patch_pairs:
        for patch in extract_patches(release, patch_size=patch_size, stride=patch_stride):
            samples.append((patch, biofilm_label))
            samples.append((rotate_image_90(patch),  biofilm_label))
            samples.append((rotate_image_180(patch), biofilm_label))
            samples.append((rotate_image_270(patch), biofilm_label))
    return samples


# -----------------------------
# Public: split by original image FIRST, then build/augment per split
# -----------------------------
def get_dataloaders(root, cfg):
    """
    Build train/test DataLoaders with leakage-free split:
    split on original images first, then patch/augment within each split.
    """
    # load paired raw images
    biofilm_dir = f"{root}/biofilm"
    release_dir = f"{root}/release"
    biofilm_images = load_images(biofilm_dir)
    release_images = load_images(release_dir)
    raw_pairs = list(zip(biofilm_images, release_images))  # [(biofilm_img, release_img), ...]

    # train/test split at image level (pre-augmentation)
    train_raw, test_raw = train_test_split(
        raw_pairs,
        train_size=0.9,
        random_state=42,
        shuffle=True,
    )

    train_raw, validation_raw = train_test_split(
        train_raw,
        train_size=0.8,
        random_state=42,
        shuffle=True,
    )

    # build (img,label) samples for each split
    train_samples = _build_pairs(
        raw_pairs=train_raw,
        threshold_method=cfg["threshold_method"],
        patch_size=cfg["patch_size"],
        patch_stride=cfg["patch_stride"],
    )
    validation_samples = _build_pairs(
        raw_pairs=validation_raw,
        threshold_method=cfg["threshold_method"],
        patch_size=cfg["patch_size"],
        patch_stride=cfg["patch_stride"],
    )

    # wrap in ImageLabelDataset
    train_samples = ImageLabelDataset(train_samples)
    validation_samples = ImageLabelDataset(validation_samples)

    # create DataLoaders
    train_loader = DataLoader(
        train_samples,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_samples,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, validation_loader


if __name__ == "__main__":
    train_loader, validation_loader = get_dataloaders(
        root="raw_data_reorganized",
        cfg={
            "batch_size": 32,
            "patch_size": 128,
            "patch_stride": 128,
            "threshold_method": "iterative",
        },
    )
