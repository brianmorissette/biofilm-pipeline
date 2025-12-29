import sys
from pathlib import Path

# Add src to path so imports work
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_pipeline.release_preprocess import *
from data_pipeline.biofilm_preprocess import *
from utils import *

THRESHOLD_METHOD = "iterative"
PATCH_SIZE = 128
PATCH_STRIDE = 128


biofilm_images = load_images("raw_data_reorganized/biofilm")
release_images = load_images("raw_data_reorganized/release")

raw_pairs = [
    (biofilm, release) for biofilm, release in zip(biofilm_images, release_images)
]
print(len(raw_pairs))
print(raw_pairs[0][0])
print(raw_pairs[0][1])


# # Show raw biofilm images before and after thresholding
# biofilm_threshold_pairs = []
# for biofilm, _ in raw_pairs:
#     preprocessed_biofilm = preprocess_biofilm(biofilm)
#     threshold = threshold_image(preprocessed_biofilm, threshold_method="iterative")
#     mask_biofilm = (preprocessed_biofilm > threshold).astype(np.float32)
#     biofilm_threshold_pairs.append((biofilm, mask_biofilm))

# display_image_pairs(biofilm_threshold_pairs, num_pairs=12, pairs_per_row=3)


pre_patch_and_rotate_pairs = []
for biofilm, release in raw_pairs:
    # preprocess release
    grayscale_release = grayscale(release)
    normalized_release = normalize(grayscale_release)

    # preprocess biofilm
    preprocessed_biofilm = preprocess_biofilm(biofilm)
    threshold = threshold_image(preprocessed_biofilm, threshold_method=THRESHOLD_METHOD)
    biofilm_label = get_biofilm_label(
        preprocessed_biofilm, threshold, label="surface area"
    )

    # create pair
    pre_patch_and_rotate_pairs.append((normalized_release, biofilm_label))

print(len(pre_patch_and_rotate_pairs))
print(pre_patch_and_rotate_pairs[0][0])
print(pre_patch_and_rotate_pairs[0][1])

post_patch_and_rotate_pairs = []
for release, biofilm_label in pre_patch_and_rotate_pairs:
    patches = extract_patches(release, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
    for patch in patches:
        rotated_patch_90 = rotate_image_90(patch)
        rotated_patch_180 = rotate_image_180(patch)
        rotated_patch_270 = rotate_image_270(patch)
        post_patch_and_rotate_pairs.append((rotated_patch_90, biofilm_label))
        post_patch_and_rotate_pairs.append((rotated_patch_180, biofilm_label))
        post_patch_and_rotate_pairs.append((rotated_patch_270, biofilm_label))

print(len(post_patch_and_rotate_pairs))
print(post_patch_and_rotate_pairs[0][0])
print(post_patch_and_rotate_pairs[0][1])

patches = []
for release, biofilm_label in post_patch_and_rotate_pairs[:20]:
    patches.append(release)
display_grid_of_images(patches)
print(type(post_patch_and_rotate_pairs[0][0]))
print(type(post_patch_and_rotate_pairs[0][1]))
print(post_patch_and_rotate_pairs[0][0].shape)
print(post_patch_and_rotate_pairs[0][1].shape)
