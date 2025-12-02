import sys
from pathlib import Path

# Add src to sys.path so imports inside build_dataset work
sys.path.append(str(Path(__file__).parent / "src"))
# sys.path.append(str(Path(__file__).parent / "data"))

import matplotlib.pyplot as plt
import numpy as np
from data_pipeline.transforms import fft_dct
from data_pipeline.build_dataset import get_dataloaders

def test_fft_dct():
    print("Starting test...")
    # Get dataloaders to access a sample image
    # Note: root path adjusted to be relative to where this script runs (project root)
    try:
        train_loader, _ = get_dataloaders(
            root="data/raw_data_reorganized", 
            cfg={
                "batch_size": 1, 
                "patch_method": "robust",
                "patch_size": 128,
                "stride_multiplier": 1,
                "threshold_method": "iterative"
            }
        )
    except Exception as e:
        print(f"Error getting dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Dataloaders created.")
    if len(train_loader.dataset) == 0:
        print("Train dataset is empty!")
        return

    # Get the first sample from the dataset
    # sample is (image_tensor, label)
    image_tensor, label = train_loader.dataset[0]
    
    # Convert tensor back to numpy for display and transform
    # The tensor is (C, H, W), we want (H, W) for a single channel grayscale image
    original_image = image_tensor.squeeze().numpy()
    
    # Apply the transform
    transformed_image = fft_dct(original_image)
    
    # Display before and after
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original Image
    im1 = axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Release Image")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Transformed Image (Log scale for better visibility of DCT coefficients)
    # Using log scale because DCT coefficients usually have a very high dynamic range
    im2 = axes[1].imshow(np.log(np.abs(transformed_image) + 1e-9), cmap='viridis')
    axes[1].set_title("DCT Transformed (Log Scale)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Original shape: {original_image.shape}")
    print(f"Transformed shape: {transformed_image.shape}")
    print(f"Original range: [{original_image.min():.4f}, {original_image.max():.4f}]")
    print(f"Transformed range: [{transformed_image.min():.4f}, {transformed_image.max():.4f}]")

if __name__ == "__main__":
    test_fft_dct()
