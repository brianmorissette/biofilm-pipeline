from data_loading.release_preprocess import *
from data_loading.biofilm_preprocess import *
from utils import *


biofilm_images = load_images("biofilm_data/biofilm")
release_images = load_images("biofilm_data/release_cells")

biofilm_preprocessed_images = []
for image in biofilm_images:
    preprocessed_image = preprocess_biofilm(image)
    biofilm_preprocessed_images.append(preprocessed_image)

display_grid_of_images(biofilm_preprocessed_images)

biofilm_labels = []
for image in biofilm_preprocessed_images:
    threshold = threshold_image(image, threshold_method="iterative")
    biofilm_label = get_biofilm_label(image, threshold, label="surface area")
    biofilm_labels.append(biofilm_label)
    print(f"Biofilm label: {biofilm_label}")

release_patches = []
for image in release_images:
    image = grayscale(image)
    image = normalize(image)
    patches = extract_patches(image, patch_size=28, stride=28)
    release_patches.append(patches)
        
display_grid_of_images(release_patches[0])  # Display patches from the first release image