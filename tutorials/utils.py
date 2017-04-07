"""

    Utilities for Tutorials
    ~~~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow


def show_image(image, cmap='gray'):
    if 'PIL' in type(image).__name__ or 'Image' in type(image).__name__:
        loaded_image = np.array(image)
    elif isinstance(image, np.ndarray):
        loaded_image = image
    elif isinstance(image, str):
        loaded_image = np.array(Image.open(image))
    else:
        raise TypeError("`image` is of an unrecognized type.")
    return imshow(loaded_image, cmap=cmap)
