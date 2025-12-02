import numpy as np
from skimage.util import view_as_windows
from skimage import exposure

def extract_patches(image, patch_size, stride_multiplier):
    """
    Extract patches from an image.
    By default, patches are non-overlapping (stride=patch_size).
    """
    patches = []
    h, w = image.shape
    stride = int(patch_size * stride_multiplier)
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def extract_patches_robust(image, patch_size, stride_multiplier, pad_mode='reflect'):
    """
    Robustly extracts patches with padding to ensure edges are kept.
    
    Args:
        image (np.array): Input image (2D).
        patch_size (tuple): Height and width of patch (h, w).
        stride (int): Step size (smaller stride = more overlap = more data).
        pad_mode (str): 'reflect', 'constant', etc. 'reflect' is best for bio-images
                        to avoid artificial black borders.
    
    Returns:
        np.array: Array of patches (N, patch_h, patch_w).
    """
    h, w = image.shape
    stride = int(patch_size * stride_multiplier)
    
    # 1. Calculate required padding to ensure the window slides to the very edge
    # We want to ensure: (padded_dim - patch_dim) % stride == 0
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    
    # Apply padding (symmetric usually keeps the center centered)
    # We add padding to the bottom and right to complete the grid
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode=pad_mode)
    
    # 2. Extract windows using efficient array views
    # Output shape: (n_rows, n_cols, patch_h, patch_w)
    # Note: view_as_windows expects window_shape to be tuple if input is > 1D
    if isinstance(patch_size, int):
        patch_shape = (patch_size, patch_size)
    else:
        patch_shape = patch_size
        
    windows = view_as_windows(image_padded, patch_shape, step=stride)
    
    # 3. Flatten into a list of patches: (N, patch_h, patch_w)
    patches = windows.reshape(-1, patch_shape[0], patch_shape[1])
    
    return patches

# Example Usage:
# Assuming 'biofilm_img' is your loaded 480x640 numpy array
# patches = create_patches_robust(biofilm_img, patch_size=(64, 64), stride=16)
# print(f"Generated {patches.shape[0]} patches.")