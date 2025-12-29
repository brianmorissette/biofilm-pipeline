import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path so imports work
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_pipeline.release_preprocess import extract_patches, extract_patches_robust, extract_patches_auto
from utils import load_images, grayscale, normalize_image, display_grid_of_images, display_image

def test_patching_strategies():
    # Load images
    # Try different paths where data might be
    possible_paths = [
        Path("data/raw_data_reorganized/release"),
        Path("raw_data_reorganized/release")
    ]
    
    image_dir = None
    for p in possible_paths:
        if p.exists():
            image_dir = p
            break
            
    if image_dir is None:
        print("Error: Could not find release images directory.")
        return
    
    print(f"Loading images from {image_dir}...")
    images = load_images(image_dir)
    if not images:
        print("No images found.")
        return

    # Use the first image
    original_image = images[0]
    
    # Preprocess
    # Release images are typically green channel (index 1) in this dataset
    gray_image = grayscale(original_image)
    norm_image = normalize_image(gray_image)
    
    print(f"Image shape: {original_image.shape}")
    print(f"Normalized shape: {norm_image.shape}")

    # Strategies to test
    strategies = [
        ("Auto Patching (25% Overlap)", extract_patches_auto, {"patch_size": 160, "target_overlap": 0.25}),
        ("Auto Patching (10% Overlap)", extract_patches_auto, {"patch_size": 160, "target_overlap": 0.1}),
        ("Auto Patching (0% Overlap)", extract_patches_auto, {"patch_size": 160, "target_overlap": 0.0}),
        ("Auto Patching (0% Overlap)", extract_patches_auto, {"patch_size": 80, "target_overlap": 0.25}),
        ("Auto Patching (0% Overlap)", extract_patches_auto, {"patch_size": 80, "target_overlap": 0.1}),
        ("Auto Patching (0% Overlap)", extract_patches_auto, {"patch_size": 80, "target_overlap": 0.0}),
    ]
    
    for name, func, kwargs in strategies:
        print(f"\n--- Testing: {name} ---")
        patches = func(norm_image, **kwargs)
        print(f"Extracted {len(patches)} patches.")
        
        # Display Original
        print("Displaying Original Image...")
        plt.figure(figsize=(6, 6))
        plt.title(f"Original Image (for {name})")
        plt.imshow(norm_image, cmap='gray')
        plt.axis("off")
        plt.show()
        
        # Display Patches
        print("Displaying Patches Grid...")
        # display_grid_of_images creates its own figure and calls show()
        display_grid_of_images(patches)

if __name__ == "__main__":
    test_patching_strategies()
