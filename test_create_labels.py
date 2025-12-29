import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.utils import load_images, display_image
from src.data_pipeline.biofilm_preprocess import (
    preprocess_biofilm,
    threshold_image,
    get_biofilm_label,
)


def test_create_labels():
    """
    Loads a few biofilm images, processes them, calculates surface area,
    and displays the results to help visualize the scale of the labels.
    """

    # 1. Setup paths
    project_root = Path(__file__).parent
    data_root = project_root / "data" / "raw_data_reorganized" / "biofilm"

    print(f"Looking for images in: {data_root}")

    # 2. Load images
    images = load_images(data_root)
    if not images:
        print("No images found! Check the path.")
        return

    # Select a few random images to inspect (e.g., 3 images)
    num_to_show = 3
    indices = np.random.choice(
        len(images), min(len(images), num_to_show), replace=False
    )

    selected_images = [images[i] for i in indices]

    # 3. Process and Calculate Labels
    print(f"\n--- Analyzing {len(selected_images)} Random Biofilm Images ---\n")

    for i, original_img in enumerate(selected_images):
        print(f"Image {i+1}:")

        # A. Preprocessing
        processed_img = preprocess_biofilm(original_img)

        # B. Thresholding
        threshold_val = threshold_image(processed_img, threshold_method="iterative")
        print(f"  - Threshold Value: {threshold_val}")

        # C. Calculate Surface Area (Label)
        surface_area = get_biofilm_label(
            processed_img, threshold_val, label="surface area"
        )
        print(f"  - Surface Area: {surface_area:.2f} square microns")

        # D. Visualization
        # Show original, processed, and binary mask side-by-side
        binary_mask = (processed_img > threshold_val).astype(np.uint8) * 255

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original (convert BGR to RGB for matplotlib if needed)
        if original_img.ndim == 3:
            original_show = original_img[:, :, ::-1]  # BGR to RGB
        else:
            original_show = original_img

        axes[0].imshow(original_show)
        axes[0].set_title("Original Biofilm")
        axes[0].axis("off")

        axes[1].imshow(processed_img, cmap="gray")
        axes[1].set_title("Preprocessed")
        axes[1].axis("off")

        axes[2].imshow(binary_mask, cmap="gray")
        axes[2].set_title(f"Mask (Area: {surface_area:.0f} $\mu m^2$)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        print("-" * 50)


if __name__ == "__main__":
    test_create_labels()
