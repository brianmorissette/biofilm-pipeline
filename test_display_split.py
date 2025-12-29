import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# Add src to sys.path so imports work
sys.path.append(str(Path(__file__).parent / "src"))

from utils import load_images, display_image_pairs


def main():
    print("Loading images...")

    # Define paths (adjust if your data structure is different)
    # Assuming data/raw_data_reorganized exists from previous context
    data_root = Path("data/raw_data_reorganized")
    if not data_root.exists():
        print(f"Error: {data_root} does not exist.")
        return

    biofilm_dir = data_root / "biofilm"
    release_dir = data_root / "release"

    # Load images
    biofilm_images = load_images(biofilm_dir)
    release_images = load_images(release_dir)

    print(
        f"Loaded {len(biofilm_images)} biofilm images and {len(release_images)} release images."
    )

    if len(biofilm_images) != len(release_images):
        print("Warning: Number of images does not match! Truncating to shorter length.")
        min_len = min(len(biofilm_images), len(release_images))
        biofilm_images = biofilm_images[:min_len]
        release_images = release_images[:min_len]

    # Create pairs
    pairs = list(zip(biofilm_images, release_images))

    # 70/20/10 Split
    # First split: 70% Train, 30% Temp (Val + Test)
    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.3, random_state=42)

    # Second split: Split the 30% Temp into 2/3 Validation (20% total) and 1/3 Test (10% total)
    # 0.2 / 0.3 = 0.66... -> Validation size in temp split
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=1 / 3, random_state=42
    )

    print(f"\nSplit Sizes:")
    print(f"Train: {len(train_pairs)} ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"Validation: {len(val_pairs)} ({len(val_pairs)/len(pairs)*100:.1f}%)")
    print(f"Test: {len(test_pairs)} ({len(test_pairs)/len(pairs)*100:.1f}%)")

    print("\nDisplaying Train Set Pairs...")
    display_image_pairs(train_pairs, num_pairs=min(10, len(train_pairs)))

    print("\nDisplaying Validation Set Pairs...")
    display_image_pairs(val_pairs, num_pairs=min(10, len(val_pairs)))

    print("\nDisplaying Test Set Pairs...")
    display_image_pairs(test_pairs, num_pairs=min(10, len(test_pairs)))


if __name__ == "__main__":
    main()
