import sys
from pathlib import Path

# Add src to path so imports work
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_pipeline.release_preprocess import *
from data_pipeline.biofilm_preprocess import *
from utils import *
import matplotlib.pyplot as plt

PATCH_SIZE = 128
PATCH_STRIDE_MULTIPLIER = 0.75

release_image = load_images("raw_data_reorganized/release")[0]
print(f"Release image shape: {release_image.shape}")
grayscale_release = grayscale(release_image)
normalized_release = normalize(grayscale_release)
patches = extract_patches_robust(normalized_release, patch_size=PATCH_SIZE, stride_multiplier=PATCH_STRIDE_MULTIPLIER)




# plot the normalized release image and the patches grid side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Display the normalized_release image
axes[0].imshow(normalized_release)
axes[0].set_title("Normalized Release Image")
axes[0].axis('off')

# Display the patches as a grid in axes[1]
num_patches = len(patches)
grid_size = int(np.ceil(np.sqrt(num_patches)))
for i in range(num_patches):
    row = i // grid_size
    col = i % grid_size
    # Create a subplot in the right axes using inset axes
    ax_inset = axes[1].inset_axes([
        (col / grid_size),
        1 - ((row + 1) / grid_size),
        1 / grid_size,
        1 / grid_size
    ])
    ax_inset.imshow(patches[i])
    ax_inset.axis('off')

axes[1].set_title("Patches Grid")
axes[1].axis('off')
plt.tight_layout()
plt.show()








