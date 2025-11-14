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