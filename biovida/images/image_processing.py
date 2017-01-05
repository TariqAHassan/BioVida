"""

    Image Processing
    ~~~~~~~~~~~~~~~~


"""
# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from biovida.images.openi_interface import OpenInterface
from biovida.support_tools.printing import pandas_pretty_print


# Goals:
#     Drop:
#       - grids (perhaps split instead?) *
#       - images with arrows *
#
#     Clean:
#       - MedPix(R) logo *
#       - all other text *
#
#     Mark:
#       - whether or an image is grayscale     X
#
# Legend:
#     * = will require a neural net.
#     p = partially solved
#     X = Solved

# ------------------------------------------------------------------------------------------
# Get sample data
# ------------------------------------------------------------------------------------------


# Create an instance of the Open-i data harvesting tool
opi = OpenInterface(n_bounds_limit=5)

# Get the Data
opi.search(None, image_type=['mri'])
df = opi.pull()

# Path to the data
raw_img_path = opi._created_img_dirs['raw']


# ------------------------------------------------------------------------------------------
# Grayscale Analysis
# ------------------------------------------------------------------------------------------


def grayscale_img(path):
    """

    Use the PIL library to determine whether or not an image is grayscale.
    Note: this tool is very conservative. *Any* 'color' will yeild `False`.

    See: http://stackoverflow.com/q/23660929/4898004
    :param path:
    :return: ``True`` if grayscale, else ``False``.
    :rtype: bool
    """
    if path is None or pd.isnull(path):
        return np.NaN
    stat = ImageStat.Stat(Image.open(path).convert("RGB"))
    return np.mean(stat.sum) == stat.sum[0]

# Compute whether or not the image is grayscale
df['grayscale_img'] = df['img_cache_name'].map(lambda i: grayscale_img(os.path.join(raw_img_path, i)))


# ------------------------------------------------------------------------------------------
# Logo Removal
# ------------------------------------------------------------------------------------------


















































