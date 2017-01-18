"""

    General Tools for Image Processing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np
from PIL import Image
from scipy.misc import imread, imresize
from keras.preprocessing import image
from skimage.color.colorconv import rgb2gray


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

    :param converted_image:
    :param img_size:
    :param axes:
    :return:
    """
    return np.transpose(imresize(converted_image, img_size), axes).astype('float32')


def load_and_scale_imgs(list_of_images, img_size, axes=(2, 0, 1), status=None):
    """

    :param list_of_images:
    :param img_size:
    :param axes:
    :param status:
    :return:
    """
    # Source: https://blog.rescale.com/neural-networks-using-keras-on-rescale/
    def status_bar(x):
        return x if status == None else status(x)

    def load_func(img):
        if 'ndarray' in str(type(cropped_image)):
            converted_image = img
        else:
            # Grayscale images by first converting them to RGB (otherwise, `imresize()` will break).
            converted_image = np.asarray(Image.open(img).convert("RGB"))
        return image_transposer(converted_image, img_size, axes=axes)

    return np.array([load_func(img_name) for img_name in status_bar(list_of_images)]) / 255.0


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





































































































































