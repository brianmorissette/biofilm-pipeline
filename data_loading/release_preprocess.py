def extract_patches(image, patch_size, stride=None):
    """
    Extract patches from an image.
    By default, patches are non-overlapping (stride=patch_size).
    """
    if stride is None:
        stride = patch_size
    patches = []
    h, w = image.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches