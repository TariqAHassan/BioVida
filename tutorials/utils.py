"""

    Utilities for Tutorials
    ~~~~~~~~~~~~~~~~~~~~~~~

"""
import qgrid  # pip3 install git+https://github.com/quantopian/qgrid
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow


def qshow(data_frame, column_width=200):
    qgrid.show_grid(data_frame, grid_options={'forceFitColumns': False, 'defaultColumnWidth': column_width})


def show_image(image):
    if 'PIL' in type(image).__name__ or 'Image' in type(image).__name__:
        loaded_image = np.array(image)
    elif isinstance(image, np.ndarray):
        loaded_image = image
    elif isinstance(image, str):
        loaded_image = np.array(Image.open(image))
    else:
        raise TypeError("`image` is of an unrecognized type.")
    return imshow(loaded_image)
