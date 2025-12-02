import scipy.fft
import numpy as np

def get_transform(image, transform_name):
    if transform_name == "none":
        return image
    elif transform_name == "fft_dct":
        return fft_dct(image)
    elif transform_name == "mexican_hat":
        return mexican_hat_transform(image)
    else:
        raise ValueError(f"Invalid transform: {transform_name}")

def fft_dct(image):
    dct_image = scipy.fft.dctn(image, type=2, norm='ortho')
    return dct_image

def mexican_hat_function(size=21, sigma=3.0):
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2
    kernel = (1 - r2 / (2*sigma**2)) * np.exp(-r2 / (2*sigma**2))
    return kernel / (kernel.sum() if kernel.sum() != 0 else 1.0)

# Function to apply Mexican Hat transform to an image
def mexican_hat_transform(image):
    kernel = mexican_hat_function(size=21, sigma=3.0)
    transformed_image = scipy.ndimage.convolve(image, kernel, mode='reflect')
    return transformed_image
    



