"""
Display biofilm images alongside their preprocessed+thresholded versions.

This script loads biofilm images, pairs them by key (treatment_date_xy) with
release for pairing consistency, and displays: left = raw biofilm, right = same
biofilm after preprocessing and thresholding (for comparison).
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from data_pipeline.biofilm_preprocess import (
    preprocess_biofilm,
    threshold_image,
)


def load_images_with_names(root):
    """
    Load all .tif images from a directory and return them with their filenames.

    Args:
        root: Path to directory containing images.

    Returns:
        List of tuples: (filename_stem, image_array)
    """
    root_path = Path(root)
    paths = sorted(root_path.glob("*.tif"), key=lambda p: p.name.casefold())
    
    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append((p.stem, img))
        else:
            print(f"Warning: Could not load {p}")
    
    return images


def extract_key(filename, prefix):
    """
    Extract the identifying key from a filename by removing the prefix.
    
    Example: 
        'biofilm_DNaseI_06-08-2025_7' -> 'DNaseI_06-08-2025_7'
        'release_DNaseI_06-08-2025_7' -> 'DNaseI_06-08-2025_7'
    
    Args:
        filename: The filename stem (without extension).
        prefix: The prefix to remove ('biofilm_' or 'release_').
    
    Returns:
        The key identifying this image pair.
    """
    if filename.startswith(prefix):
        return filename[len(prefix):]
    return filename


def display_all_pairs(
    data_root,
    pairs_per_row=4,
    max_rows_per_page=4,
    figsize_per_pair=(4, 2),
):
    """
    Display all biofilm/release pairs in a grid.
    
    Args:
        data_root: Path to the data directory containing 'biofilm' and 'release' folders.
        pairs_per_row: Number of pairs to display per row.
        figsize_per_pair: (width, height) in inches for each pair.
    """
    data_path = Path(data_root)
    biofilm_dir = data_path / "biofilm"
    release_dir = data_path / "release"
    
    if not biofilm_dir.exists():
        print(f"Error: Biofilm directory not found: {biofilm_dir}")
        return
    if not release_dir.exists():
        print(f"Error: Release directory not found: {release_dir}")
        return
    
    # Load images
    print("Loading biofilm images...")
    biofilm_images = load_images_with_names(biofilm_dir)
    print(f"  Loaded {len(biofilm_images)} biofilm images")
    
    print("Loading release image names (for pairing only)...")
    release_images = load_images_with_names(release_dir)
    print(f"  Loaded {len(release_images)} release images")
    
    # Create dictionaries keyed by the identifying part
    biofilm_dict = {extract_key(name, "biofilm_"): img for name, img in biofilm_images}
    release_dict = {extract_key(name, "release_"): img for name, img in release_images}
    
    THRESHOLD_METHODS = ("iterative", "adaptive", "otsu")

    # Find all keys and identify matches/mismatches
    all_keys = sorted(set(biofilm_dict.keys()) | set(release_dict.keys()))
    
    paired_keys = []
    biofilm_only = []
    release_only = []
    
    for key in all_keys:
        has_biofilm = key in biofilm_dict
        has_release = key in release_dict
        
        if has_biofilm and has_release:
            paired_keys.append(key)
        elif has_biofilm:
            biofilm_only.append(key)
        else:
            release_only.append(key)
    
    # Report statistics
    print(f"\n=== Pairing Statistics ===")
    print(f"Complete pairs: {len(paired_keys)}")
    print(f"Biofilm only (missing release): {len(biofilm_only)}")
    print(f"Release only (missing biofilm): {len(release_only)}")
    
    if biofilm_only:
        print(f"\nBiofilm images missing release pair:")
        for key in biofilm_only[:10]:  # Show first 10
            print(f"  - {key}")
        if len(biofilm_only) > 10:
            print(f"  ... and {len(biofilm_only) - 10} more")
    
    if release_only:
        print(f"\nRelease images missing biofilm pair:")
        for key in release_only[:10]:  # Show first 10
            print(f"  - {key}")
        if len(release_only) > 10:
            print(f"  ... and {len(release_only) - 10} more")
    
    if not paired_keys:
        print("\nNo complete pairs found! Check your data directory.")
        return
    
    # Paginate the display so each figure fits on screen
    num_pairs = len(paired_keys)
    max_pairs_per_page = pairs_per_row * max_rows_per_page
    num_pages = (num_pairs + max_pairs_per_page - 1) // max_pairs_per_page

    print(
        f"\n=== Displaying {num_pairs} pairs "
        f"over {num_pages} page(s), up to {max_pairs_per_page} pairs per page ==="
    )
    print("Close each figure window to advance to the next page.\n")

    for page_idx in range(num_pages):
        start = page_idx * max_pairs_per_page
        end = min(start + max_pairs_per_page, num_pairs)
        page_keys = paired_keys[start:end]

        page_pairs = len(page_keys)
        num_rows = (page_pairs + pairs_per_row - 1) // pairs_per_row
        # Columns: raw + iterative + adaptive + otsu
        num_cols = pairs_per_row * 4

        fig_width = figsize_per_pair[0] * num_cols
        fig_height = figsize_per_pair[1] * num_rows

        print(
            f"Page {page_idx + 1}/{num_pages}: "
            f"showing pairs {start + 1}–{end} "
            f"in {num_rows} row(s), figure size "
            f"{fig_width:.1f} x {fig_height:.1f} inches"
        )

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(fig_width, fig_height)
        )

        if num_rows == 1:
            axes = axes.reshape(1, -1)

        for ax in axes.flat:
            ax.axis("off")

        for i, key in enumerate(page_keys):
            row = i // pairs_per_row
            col_offset = (i % pairs_per_row) * 4

            biofilm_img = biofilm_dict[key]

            if biofilm_img.ndim == 3 and biofilm_img.shape[2] >= 3:
                biofilm_display = cv2.cvtColor(
                    biofilm_img, cv2.COLOR_BGR2RGB
                )
            else:
                biofilm_display = biofilm_img

            preprocessed = preprocess_biofilm(biofilm_img)

            # Raw biofilm
            axes[row, col_offset].imshow(biofilm_display)
            axes[row, col_offset].set_title(f"Raw: {key}", fontsize=6)

            # Iterative, adaptive, Otsu
            for j, method in enumerate(THRESHOLD_METHODS):
                result = threshold_image(preprocessed, threshold_method=method)
                if isinstance(result, np.ndarray):
                    processed_display = result
                else:
                    processed_display = (
                        (preprocessed > result).astype(np.uint8) * 255
                    )
                axes[row, col_offset + 1 + j].imshow(
                    processed_display, cmap="gray"
                )
                axes[row, col_offset + 1 + j].set_title(
                    f"{method}: {key}", fontsize=6
                )

        plt.suptitle(
            f"Biofilm: Raw vs Iterative / Adaptive / Otsu thresholding\n"
            f"Page {page_idx + 1}/{num_pages} "
            f"({start + 1}–{end} of {num_pairs} pairs)",
            fontsize=10,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()


def display_unpaired_images(data_root, images_per_row=6, figsize_per_image=(2, 2)):
    """
    Display any unpaired images (biofilm without release or vice versa).
    
    Args:
        data_root: Path to the data directory.
        images_per_row: Number of images per row.
        figsize_per_image: (width, height) in inches for each image.
    """
    data_path = Path(data_root)
    biofilm_dir = data_path / "biofilm"
    release_dir = data_path / "release"
    
    # Load images
    biofilm_images = load_images_with_names(biofilm_dir)
    release_images = load_images_with_names(release_dir)
    
    # Create dictionaries
    biofilm_dict = {extract_key(name, "biofilm_"): (name, img) for name, img in biofilm_images}
    release_dict = {extract_key(name, "release_"): (name, img) for name, img in release_images}
    
    # Find unpaired
    biofilm_only = [(key, biofilm_dict[key]) for key in biofilm_dict if key not in release_dict]
    release_only = [(key, release_dict[key]) for key in release_dict if key not in biofilm_dict]
    
    # Display biofilm-only images
    if biofilm_only:
        print(f"\n=== Displaying {len(biofilm_only)} unpaired BIOFILM images ===")
        _display_image_grid(
            [(name, img) for key, (name, img) in biofilm_only],
            title="Unpaired Biofilm Images (Missing Release)",
            images_per_row=images_per_row,
            figsize_per_image=figsize_per_image
        )
    
    # Display release-only images
    if release_only:
        print(f"\n=== Displaying {len(release_only)} unpaired RELEASE images ===")
        _display_image_grid(
            [(name, img) for key, (name, img) in release_only],
            title="Unpaired Release Images (Missing Biofilm)",
            images_per_row=images_per_row,
            figsize_per_image=figsize_per_image
        )
    
    if not biofilm_only and not release_only:
        print("\nAll images are properly paired!")


def _display_image_grid(images, title, images_per_row=6, figsize_per_image=(2, 2)):
    """Helper to display a grid of images."""
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    fig_width = figsize_per_image[0] * images_per_row
    fig_height = figsize_per_image[1] * num_rows
    
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(fig_width, fig_height))
    
    if num_rows == 1 and images_per_row == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif images_per_row == 1:
        axes = axes.reshape(-1, 1)
    
    for ax in axes.flat:
        ax.axis("off")
    
    for i, (name, img) in enumerate(images):
        row = i // images_per_row
        col = i % images_per_row
        
        if img.ndim == 3 and img.shape[2] >= 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            display_img = img
        
        axes[row, col].imshow(display_img)
        axes[row, col].set_title(name, fontsize=5)
    
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Default data path
    data_root = Path(__file__).parent / "data" / "raw_data_reorganized_new"
    
    print(f"Data directory: {data_root}")
    print("=" * 60)
    
    # Display all paired images in manageable pages
    display_all_pairs(
        data_root,
        pairs_per_row=4,
        max_rows_per_page=4,
        figsize_per_pair=(3, 2),
    )
    
    # Optionally display unpaired images
    display_unpaired_images(data_root)
