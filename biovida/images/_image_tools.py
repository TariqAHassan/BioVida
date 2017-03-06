"""

    General Tools for the Image Subpackage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from time import sleep
from warnings import warn
from scipy.misc import imread, imresize
from skimage.color.colorconv import rgb2gray

from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import multimap
from biovida.support_tools.support_tools import items_null


# ----------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------


TIME_FORMAT = "%Y_%h_%d_%H_%M_%S_%f"


# ----------------------------------------------------------------------------------------------------------
# General Tools
# ----------------------------------------------------------------------------------------------------------


class NoResultsFound(Exception):
    pass


def sleep_with_noise(amount_of_time, mean=0.0, noise=0.75):
    """

    Sleep the current python instance by `amount_of_time`.

    :param amount_of_time: the amount of time to sleep the instance for.
    :type amount_of_time: ``int`` or ``float``
    :param mean: ``mean`` param of ``numpy.random.normal()``. Defaults to 0.0.
    :type mean: ``float``
    :param noise: ``scale`` param of ``numpy.random.normal()``. Defaults to 0.75.
    :type noise: ``float``
    """
    sleep(abs(amount_of_time + np.random.normal(loc=mean, scale=noise)))


def try_fuzzywuzzy_import():
    """

    Try to import the ``fuzzywuzzy`` library.

    """
    try:
        from fuzzywuzzy import process
        return process
    except ImportError:
        error_msg = "`fuzzy_threshold` requires the `fuzzywuzzy` library,\n" \
                    "which can be installed with `$ pip install fuzzywuzzy`.\n" \
                    "For best performance, it is also recommended that python-Levenshtein is installed.\n" \
                    "(`pip install python-levenshtein`)."
        raise ImportError(error_msg)


def resetting_label(to_label):
    """

    Label repeats in a list.

    :param to_label: a list with repeating elements.
    :type to_label: ``tuple`` or ``list``
    :return: a list of string of the form shown in the example below.
    :rtype: ``list``

    :Example:

    >>> resetting_label(['a', 'a', 'a', 'b', 'b', 'z', 'a'])
    ...
    ['a_1', 'a_2', 'a_3', 'b_1', 'b_2', 'z_1', 'a_4']

    """
    def formatted_label(existing_name, label):
        """Joins the items in `to_label` with the label number generated below."""
        return "{0}_{1}".format(existing_name, str(label))

    all_labels = list()
    label_record = dict.fromkeys(set(filter(lambda x: isinstance(x, str), to_label)), 1)
    for i in to_label:
        if not isinstance(i, str):
            all_labels.append(i)
        else:
            all_labels.append(formatted_label(cln(i), label_record[i]))
            label_record[i] += 1

    return all_labels


def load_img_rescale(path_to_image, gray_only=False):
    """

    Loads an image, converts it to grayscale and normalizes (/255.0).

    :param path_to_image: the address of the image.
    :type path_to_image: ``str``
    :return: the image as a matrix.
    :rtype: ``ndarray``
    """
    if gray_only:
        return rgb2gray(path_to_image) / 255.0
    else:
        return rgb2gray(imread(path_to_image, flatten=True)) / 255.0


def image_transposer(converted_image, img_size, axes=(2, 0, 1)):
    """

    Tool to resize and transpose an image (about given axes).

    :param converted_image: the image as a ndarray.
    :type converted_image: ``ndarray``
    :param img_size: to size to coerse the images to be, e.g., (150, 150)
    :type img_size: ``tuple``
    :param axes: the axes to transpose the image on.
    :type axes: ``tuple``
    :return: the resized and transposed image.
    :rtype: ``ndarray``
    """
    return np.transpose(imresize(converted_image, img_size), axes).astype('float32')


def load_and_scale_imgs(list_of_images, img_size, axes=(2, 0, 1), status=True, grayscale_first=False):
    """

    Load and scale a list of images from a directory

    :param list_of_images: a list of paths to images.
    :type list_of_images: ``list`` or ``tuple``
    :param img_size: to size to coerse the images to be, e.g., (150, 150)
    :type img_size: ``tuple``
    :param axes: the axes to transpose the image on.
    :type axes: ``tuple``
    :param status: if ``True``, use `tqdm` to print progress as the load progresses.
    :type status: ``bool``
    :param grayscale_first: convert the image to grayscale first.
    :type grayscale_first: ``bool``
    :return: the images as ndarrays nested inside of another ndarray.
    :rtype: ``ndarray``
    """
    # Source: https://blog.rescale.com/neural-networks-using-keras-on-rescale/
    def status_bar(x):
        if status:
            return tqdm(x)
        else:
            return x

    def load_func(img):
        if 'ndarray' in str(type(img)):
            converted_image = img
        else:
            # Load grayscale images by first converting them to RGB (otherwise, `imresize()` will break).
            if grayscale_first:
                loaded_img = Image.open(img).convert("LA")
                loaded_img = loaded_img.convert("RGB")
            else:
                loaded_img = Image.open(img).convert("RGB")
            converted_image = np.asarray(loaded_img)
        return image_transposer(converted_image, img_size, axes=axes)

    return np.array([load_func(img_name) for img_name in status_bar(list_of_images)]) / 255.0


def show_plt(img):
    """

    Use matplotlib to display an img (which is represented as a matrix).

    :param img: an img represented as a matrix.
    :type img: ``ndarray``
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


















