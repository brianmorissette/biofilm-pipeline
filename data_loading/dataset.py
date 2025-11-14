# src/data_loading/dataset.py

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_loading.biofilm_preprocess import *
from data_loading.release_preprocess import *
from utils import *



raw_biofilm_images = load_images("biofilm_data/biofilm")
raw_release_images = load_images("biofilm_data/release_cells")

release_images = []
biofilm_labels = []
biofilm_release_pairs = []

for image in raw_biofilm_images:
    preprocessed_image = preprocess_biofilm(image)
    threshold = threshold_image(preprocessed_image, threshold_method="iterative")
    label = get_biofilm_label(preprocessed_image, threshold, label="surface area")
    biofilm_labels.append(label)

for image in raw_release_images:
    grayscale_image = grayscale(image)
    normalized_image = normalize(grayscale_image)
    release_images.append(normalized_image)

for release_image, biofilm_label in zip(release_images, biofilm_labels):
    biofilm_release_pairs.append((release_image, biofilm_label))

print(biofilm_release_pairs[0][0])
print(biofilm_release_pairs[0][1])




