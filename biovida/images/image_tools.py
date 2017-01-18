"""

    General Tools for Image Processing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np
from PIL import Image
from scipy.misc import imread, imresize
from keras.preprocessing import image
from skimage.color.colorconv import rgb2gray


def load_img_rescale(path_to_image):
    """

    Loads an image, converts it to grayscale and normalizes (/255.0).

    :param path_to_image: the address of the image.
    :type path_to_image: ``str``
    :return: the image as a matrix.
    :rtype: ``ndarray``
    """
    return rgb2gray(imread(path_to_image, flatten=True)) / 255.0


def load_and_scale_imgs(list_of_images, img_size, axes=(2, 0, 1)):
    """

    :param list_of_images:
    :param img_size:
    :param axes:
    :return:
    """
    # img = list_of_images[0]
    # Source: https://blog.rescale.com/neural-networks-using-keras-on-rescale/
    def load_func(img):
        # This step handles grayscale images by first converting them to RGB.
        # Otherwise, `imresize()` will break.
        converted_image = np.asarray(Image.open(img).convert("RGB"))
        return np.transpose(imresize(converted_image, img_size), axes).astype('float32')
    return np.array([load_func(img_name) for img_name in list_of_images]) / 255.0


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


# img = '/Users/tariq/Desktop/15_arrow.png'
# img = '/Users/tariq/Desktop/101.png'
#
# np.expand_dims(imread(img), axis=1).shape
#
# imread('/Users/tariq/Desktop/22.png').shape
#
# imresize(imread('/Users/tariq/Desktop/15_arrow.png'), img_size)
#
# m = np.asarray(Image.open('/Users/tariq/Desktop/101.png').convert("RGB")).shape























































































































































