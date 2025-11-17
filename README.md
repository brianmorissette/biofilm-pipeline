# Biofilm Pipeline

A machine learning pipeline for analyzing biofilm images and their corresponding release measurements. This project processes paired microscopy images to predict biofilm characteristics from release patterns.

## Overview

This pipeline processes two types of microscopy images:
- **Biofilm images (20x magnification)**: Show the biofilm structure and coverage
- **Release images (60x magnification)**: Show the corresponding release patterns

The goal is to train models that can predict biofilm surface area from release image patches.

## Project Structure

```
biofilm-pipeline/
├── data_pipeline/          # Data loading and preprocessing
│   ├── biofilm_preprocess.py   # Biofilm image preprocessing and thresholding
│   ├── release_preprocess.py   # Release image preprocessing and patch extraction
│   └── dataset.py              # PyTorch dataset definitions
├── models/                 # Neural network architectures
│   ├── cnn.py                  # CNN model definitions
│   └── factory.py              # Model factory
├── training/               # Training utilities
│   └── trainer.py
├── raw_data_reorganized/   # Processed data (70 paired images)
│   ├── biofilm/                # Biofilm images
│   └── release/                # Release images
├── utils.py               # Utility functions for visualization and data loading
├── organize.py            # Script to reorganize raw data into paired structure
├── final_data_pipeline.py # Complete data pipeline for model training
└── sweep_runner.py        # Hyperparameter sweep configuration
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install dependencies
uv sync

# Or manually install with pip
pip install -r pyproject.toml
```

### Requirements
- Python >= 3.12
- PyTorch >= 2.9.1
- OpenCV >= 4.12.0
- NumPy >= 2.2.6
- Matplotlib >= 3.10.7
- scikit-learn >= 1.7.2

## Data Organization

### Image Naming Convention

Images are organized with the following naming structure:
- Biofilm: `biofilm_[TREATMENT]_[DATE]_[SAMPLE].tif`
- Release: `release_[TREATMENT]_[DATE]_[SAMPLE].tif`

**Example:**
- `biofilm_DNaseI_07-02-2025_1.tif` (biofilm image)
- `release_DNaseI_07-02-2025_1.tif` (corresponding release image)

### Treatments
- DNaseI
- ProtK (Proteinase K)
- NaIO4 (Sodium periodate)
- pH10
- Untreated (control)

### Data Statistics
- **Total paired images**: 70 pairs
- **Image types**: 20x (biofilm), 60x (release)
- Only complete pairs (matching biofilm + release) are included

## Usage

### 1. Organize Raw Data

If you have raw data in the nested folder structure, reorganize it into the paired format:

```python
python organize.py
```

This will create `raw_data_reorganized/` with separate `biofilm/` and `release/` folders containing only complete pairs.

### 2. Visualize Image Pairs

```python
from utils import load_images, display_image_pairs

# Load images
biofilm_images = load_images("raw_data_reorganized/biofilm")
release_images = load_images("raw_data_reorganized/release")

# Create pairs
pairs = [(b, r) for b, r in zip(biofilm_images, release_images)]

# Display first 5 pairs
display_image_pairs(pairs, num_pairs=5, pairs_per_row=3)
```

### 3. Preprocess Images

```python
from data_pipeline.biofilm_preprocess import preprocess_biofilm, threshold_image
from data_pipeline.release_preprocess import extract_patches
from utils import grayscale, normalize

# Preprocess biofilm
biofilm_processed = preprocess_biofilm(biofilm_image)
threshold = threshold_image(biofilm_processed, threshold_method="iterative")
surface_area = get_biofilm_label(biofilm_processed, threshold, label="surface area")

# Preprocess release
release_gray = grayscale(release_image)
release_norm = normalize(release_gray)
patches = extract_patches(release_norm, patch_size=128, stride=128)
```

### 4. Train a Model

```python
from final_data_pipeline import get_dataloaders

# Get train/validation dataloaders
train_loader, val_loader = get_dataloaders(
    biofilm_root="raw_data_reorganized/biofilm",
    release_root="raw_data_reorganized/release",
    threshold_method="iterative",
    patch_size=128,
    patch_stride=128,
    test_split=0.2,
    batch_size=32,
    num_workers=4
)

# Train your model
for images, labels in train_loader:
    # Training loop
    pass
```

## Key Features

### Biofilm Preprocessing
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Iterative thresholding for binary mask generation
- Surface area calculation
- Gaussian blur for noise reduction

### Release Image Processing
- Grayscale conversion
- Normalization
- Patch extraction with configurable size and stride
- Multiple patches per image for data augmentation

### Data Pipeline
- Automatic train/validation splitting
- PyTorch Dataset and DataLoader integration
- Paired image validation (ensures matching biofilm/release)
- Flexible preprocessing configuration

## Visualization Tools

The `utils.py` module provides helpful visualization functions:

- `display_image_pairs()`: Show biofilm and release pairs side by side in a grid
- `display_grid_of_images()`: Display multiple images in a grid
- `display_image()`: Show a single image

## Development

### Running Tests

```bash
# Test data pipeline
uv run test_data_pipeline.py

# Test patch extraction
uv run test_patch.py
```

### Hyperparameter Sweeps

Configure sweeps in `sweep_runner.yml` and run:

```bash
uv run sweep_runner.py
```

## License

[Add your license here]

## Authors

[Add authors here]

