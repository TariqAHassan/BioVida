# coding: utf-8

"""

    General Tools for the Image Subpackage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np
from PIL import Image
from time import sleep
from scipy.misc import imread, imresize
from skimage.color.colorconv import rgb2gray

# General Support Tools
from biovida.support_tools.support_tools import tqdm


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


def load_image_rescale(path_to_image, gray_only=False):
    """

    Loads an image, converts it to grayscale and normalizes (/255.0).

    :param path_to_image: the address of the image.
    :type path_to_image: ``str``
    :param gray_only: if ``True``, simply convert to grayscale and return.
                      Otherwise, read-in, flatten and convert to grayscale. Default to ``False``.
    :type gray_only: ``bool``
    :return: the image as a matrix.
    :rtype: ``ndarray``
    """
    if gray_only:
        return rgb2gray(path_to_image) / 255.0
    else:
        return rgb2gray(imread(path_to_image, flatten=True)) / 255.0


def image_transposer(converted_image, image_size, axes=(2, 0, 1)):
    """

    Tool to resize and transpose an image (about given axes).

    :param converted_image: the image as a ndarray.
    :type converted_image: ``ndarray``
    :param image_size: to size to coerse the images to be, e.g., (150, 150)
    :type image_size: ``tuple``
    :param axes: the axes to transpose the image on.
    :type axes: ``tuple``
    :return: the resized and transposed image.
    :rtype: ``ndarray``
    """
    return np.transpose(imresize(converted_image, image_size), axes).astype('float32')


def load_and_scale_images(list_of_images, image_size, axes=(2, 0, 1), status=True, grayscale_first=False, desc=None):
    """

    Load and scale a list of images from a directory

    :param list_of_images: a list of paths to images.
    :type list_of_images: ``list`` or ``tuple``
    :param image_size: to size to coerce the images to be, e.g., (150, 150)
    :type image_size: ``tuple``
    :param axes: the axes to transpose the image on.
    :type axes: ``tuple``
    :param status: if ``True``, use `tqdm` to print progress as the load progresses.
    :type status: ``bool``
    :param grayscale_first: convert the image to grayscale first.
    :type grayscale_first: ``bool``
    :param desc: description to pass to ``tqdm``.
    :type desc: ``str`` or ``None``
    :return: the images as ndarrays nested inside of another ndarray.
    :rtype: ``ndarray``
    """
    # Source: https://blog.rescale.com/neural-networks-using-keras-on-rescale/
    def load_func(image):
        if 'ndarray' == type(image).__name__:
            converted_image = image
        else:
            # Load grayscale images by first converting them to RGB (otherwise, `imresize()` will break).
            if grayscale_first:
                loaded_image = Image.open(image).convert("LA")
                loaded_image = loaded_image.convert("RGB")
            else:
                loaded_image = Image.open(image).convert("RGB")
            converted_image = np.asarray(loaded_image)
        return image_transposer(converted_image, image_size, axes=axes)

    loaded = [load_func(image_name) for image_name in tqdm(list_of_images, desc=desc, disable=not status)]
    return np.array(loaded) / 255.0


def show_plt(image):
    """

    Use matplotlib to display an image (which is represented as a matrix).

    :param image: an image represented as a matrix.
    :type image: ``ndarray``
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()
