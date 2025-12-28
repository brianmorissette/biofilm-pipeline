import cv2
import numpy as np

def threshold_image(image, threshold_method):
    if threshold_method == "iterative":
        return iterative_threshold(image)
    else:
        raise ValueError(f"Invalid threshold method: {threshold_method}")


def get_biofilm_label(image,threshold, label):
    if label == "surface area":
        return get_surface_area(image, threshold)
    else:
        raise ValueError(f"Invalid label: {label}")


def get_surface_area(image, threshold):
    """Calculate the biofilm surface area in square microns."""
    return np.sum(image > threshold) * 1.13 * 1.13  # 1.13x1.13 is the pixel size in microns  

def normalize_labels(labels):
    """
    Normalize an array-like of labels to [0, 1].
    """
    labels = np.array(labels, dtype=np.float32)
    min_val = labels.min()
    max_val = labels.max()
    if max_val - min_val == 0:
        return np.zeros_like(labels)
    return (labels - min_val) / (max_val - min_val)

def preprocess_biofilm(image, clip_limit=2.0, tile_size=(8, 8), blur_ksize=(5, 5)):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced_image = clahe.apply(gray_image)
    normalized_image = cv2.normalize(
        src=enhanced_image, 
        dst=None, 
        alpha=0, 
        beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U)
    preprocessed_image = cv2.GaussianBlur(normalized_image, blur_ksize, 0)
    return preprocessed_image

def iterative_threshold(image):
    iteration_count = 0
    current_threshold = 127.0
    last_threshold = -1.0
    tolerance = 0.5
    while abs(current_threshold - last_threshold) > tolerance:
        iteration_count += 1
        last_threshold = current_threshold
        background_pixels = image[image <= current_threshold]
        foreground_pixels = image[image > current_threshold]
        if background_pixels.size == 0:
            mean_bg = 0.0
        else:
            mean_bg = np.mean(background_pixels)
        if foreground_pixels.size == 0:
            mean_fg = 255.0
        else:
            mean_fg = np.mean(foreground_pixels)
        current_threshold = (mean_bg + mean_fg) / 2.0
    return int(round(current_threshold))

