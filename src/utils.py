import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_images(root) -> list[np.ndarray]:
    """
    Loads all .tif images from the specified root directory and its subdirectories.

    Args:
        root: The root directory to search for images.

    Returns:
        A list of loaded images as numpy arrays.
    """
    paths = sorted([*Path(root).rglob("*.tif")], key=lambda p: p.as_posix().casefold())
    return [
        img
        for p in paths
        if (img := cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) is not None
    ]


def grayscale(image) -> np.ndarray:
    """
    Extracts the green channel (index 1) from an image, assuming it represents grayscale info.

    Args:
        image: The input image (multi-channel).

    Returns:
        The single-channel image.
    """
    return image[:, :, 1]


def normalize_image(image) -> np.ndarray:
    """
    Normalizes the image pixel values to range [0, 1].

    Args:
        image: The input image.

    Returns:
        The normalized image.
    """
    return image / np.max(image)


def display_image(image) -> None:
    """
    Displays a single image using matplotlib.

    Args:
        image: The image to display.
    """
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def display_grid_of_images(images) -> None:
    """
    Displays a grid of images.

    Args:
        images: A list of images to display.
    """
    grid_size = int(np.ceil(np.sqrt(len(images))))
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def display_image_pairs(pairs, num_pairs=None, pairs_per_row=5) -> None:
    """
    Displays pairs of biofilm and release images.

    Args:
        pairs: A list of tuples, where each tuple contains (biofilm_image, release_image).
        num_pairs: The maximum number of pairs to display. If None, displays all.
        pairs_per_row: Number of pairs to display per row.
    """
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
        ax.axis("off")

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
