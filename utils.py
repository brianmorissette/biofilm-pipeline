import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_images(root) -> list[np.ndarray]:
    paths = sorted(
        [*Path(root).rglob("*.tif")],
        key=lambda p: p.as_posix().casefold()
    )
    return [img for p in paths if (img := cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) is not None]
    
def grayscale(image) -> np.ndarray:
    return image[:,:,1]

def normalize(image) -> np.ndarray:
    return image / np.max(image)

def display_image(image) -> None:
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_grid_of_images(images) -> None:
    grid_size = int(np.ceil(np.sqrt(len(images))))
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def rotate_image_90(image) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def rotate_image_180(image) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_180)

def rotate_image_270(image) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def display_image_pairs(pairs, num_pairs=None, pairs_per_row=5) -> None:
    num_to_display = len(pairs)
    if num_pairs is not None:
        num_to_display = min(num_to_display, num_pairs)

    # Calculate grid dimensions: each pair takes 2 columns (biofilm + release)
    cols = pairs_per_row * 2
    rows = int(np.ceil(num_to_display / pairs_per_row))

    # Create figure with appropriate size
    fig, axes = plt.subplots(rows, cols, figsize=(3 * pairs_per_row, 3 * rows))

    # Flatten axes array for easier indexing
    if rows == 1 and cols == 2:
        axes = axes.reshape(1, -1)
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Hide all axes first
    for ax in axes.flat:
        ax.axis('off')

    for i in range(num_to_display):
        # Calculate position in grid
        row = i // pairs_per_row
        col_offset = (i % pairs_per_row) * 2

        biofilm_image, release_image = pairs[i]

        # Display biofilm image
        axes[row, col_offset].imshow(biofilm_image)
        axes[row, col_offset].set_title(f"B{i+1}", fontsize=8)

        # Display release image
        axes[row, col_offset + 1].imshow(release_image)
        axes[row, col_offset + 1].set_title(f"R{i+1}", fontsize=8)

    plt.tight_layout(pad=0.5)
    plt.show()


