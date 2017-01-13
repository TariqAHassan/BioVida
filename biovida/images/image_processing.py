"""

    Image Processing
    ~~~~~~~~~~~~~~~~

"""
# Imports
import os
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageStat
from PIL import ImageChops
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Tools form the image subpackage
from biovida.images.openi_interface import OpenInterface
from biovida.images.models.img_classification import ImageRecognitionCNN

# General Support Tools
from biovida.support_tools.support_tools import dict_reverse
from biovida.support_tools.printing import pandas_pretty_printer

# Tool to create required caches
from biovida.support_tools._cache_management import _package_cache_creator

# Suppress np scientific notation
np.set_printoptions(suppress=True)

# Start tqdm
tqdm.pandas("status")

# ------------------------------------------------------------------------------------------
# General procedure (for Ultrasound, X-ray, CT and MRI):
# ------------------------------------------------------------------------------------------
#
#   1. Check if grayscale                                                       x
#         - if false, try to detect a frame
#               - if frame, crop else ban
#   2. Look for MedLine(R) logo *
#         - if true, crop
#   3. Look for text bar *
#         - if true, crop
#   4. Look for arrows or boxes *
#         - if true, ban
#   5. Look for image grids *
#         - if true, ban
#   6. Look for graphs *
#         - if true, ban
#   6. Look for faces *
#         - if true, ban
#   7. Look for other text in the image *
#         - if true, ban (or find method to detect text image overlay).
#
# Legend:
#     * = requires machine learning
#     p = partially solved
#     x = solved
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# Temporary
# ------------------------------------------------------------------------------------------

from biovida.images.models.temp import data_path, cache_path

# Create the required cahces
pcc = _package_cache_creator(sub_dir='images', cache_path=cache_path, to_create=['intermediate', 'processed', 'raw'])
root_path, created_img_processing_dirs = pcc

opi = OpenInterface(cache_path)
df = opi.image_record_database.copy()

# ------------------------------------------------------------------------------------------
# Tool to display images
# ------------------------------------------------------------------------------------------

def img_text(img_path, text='', color=(220, 10, 10)):
    """

    :param img_path:
    :param text:
    :param color:
    :return:
    """
    from PIL import ImageFont, Image, ImageDraw
    font = ImageFont.load_default().font
    font = ImageFont.truetype("Verdana.ttf", 12)

    img = Image.open(img_path)

    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, color, font=font)
    draw = ImageDraw.Draw(img)

    img.show()

# ------------------------------------------------------------------------------------------
# Get sample data
# ------------------------------------------------------------------------------------------

# Create an instance of the Open-i data harvesting tool
opi = OpenInterface(cache_path, download_limit=5500, img_sleep_time=2, records_sleep_main=None)

# Get the Data
opi.search(None, image_type=['mri'])
df = opi.pull()

# Save or Load the database.
# opi.cache('image_record_database', action='save')
df = opi.current_search_dataframe

# Path to the data
raw_img_path = opi._created_img_dirs['raw']
inter_img_path = created_img_processing_dirs['intermediate']

# Create column with values of the form: PATH/image_name
df['img_cache_name_full'] = df['img_cache_name'].map(lambda i: os.path.join(raw_img_path, i), na_action='ignore')

# ------------------------------------------------------------------------------------------
# Grayscale Analysis
# ------------------------------------------------------------------------------------------

def grayscale_img(img_path):
    """

    | Tool which uses the PIL library to determine whether or not an image is grayscale.
    | Note: this tool is very conservative (*any* 'color' will yeild `False`).

    :param img_path: path to an image
    :type img_path:``str``
    :return: ``True`` if grayscale, else ``False``.
    :rtype: bool
    """
    # See: http://stackoverflow.com/q/23660929/4898004
    if img_path is None or pd.isnull(img_path):
        return np.NaN
    stat = ImageStat.Stat(Image.open(img_path).convert("RGB"))
    return np.mean(stat.sum) == stat.sum[0]

# Compute whether or not the image is grayscale
# df['grayscale_img'] = df['img_cache_name_full'].progress_map(lambda i: grayscale_img(i))

def require_grayscale_check(x):
    """

    Check if image classification as an 'mri', 'pet', 'ct' is 'ultrasound'
    is valid based on whether or not the image is grayscale.

    :param x:
    :return:
    :rtype: ``bool``
    """
    grayscale_technologies = ('mri', 'ct', 'x_ray', 'ultrasound')

    if x['image_modality_major'] in grayscale_technologies and x['grayscale_img'] is False:
        return np.NaN
    else:
        return x['image_modality_major']

# Apply check based on grayscale status. (move lower)
# df['image_modality_major'] = df.apply(require_grayscale_check, axis=1)

# df[['image_modality_major']][df['img_cache_name'].str.contains("1__PMC4390528_nihms-668294-f0002__L")]

# ------------------------------------------------------------------------------------------
# Watermark Cleaning
# ------------------------------------------------------------------------------------------

import numpy as np

from skimage import data
from skimage.io import imread
from skimage.color.colorconv import rgb2gray
from skimage.feature import match_template
from scipy.misc import imresize


def _bounds_sinvar_t_match(pattern_shape, base_shape, prop_shrink=0.1, scaling_lower_limit=0.3):
    """

    Defines a vector over which to iterate when scaling the pattern shape in _scale_invar_match_template.
    This algorithm *should* solve the general form of the problem.

    Motivation:
      a. pattern is smaller than base
          - want to known x, for P * x before P large <- TL (to large)
          - set lower bound to be percentage of TL
          - compute sequence from [lower bound, upper bound]
      b. pattern is larger than base
          - want to known x for P / x until P is suffcently small <- SS (sufficently small)
          - set upper bound to be percentage of SS.
          - compute sequence from [lower bound, upper bound]

    :param pattern_img: the pattern to search for: ndim = 2. (nrows, ncols).
    :param base_img: the image to search for the patten for. ndim = 2. (nrows, ncols).
    :param prop_shrink: defines the number of steps to take between the bounds.
    :param scaling_lower_limit: smallest acceptable bound. Set to 0.0 to disable.
    :return: ``ndarray``
    """
    # WARNING: we have IMG.shape = (ROWS, COLUMNS) = (HEIGHT, WIDTH).
    def bounds_clean(arr):
        if scaling_lower_limit >= 1:
            arr = np.append(1, arr)
        if np.min(arr) < scaling_lower_limit:
            return np.append(scaling_lower_limit, arr[arr > scaling_lower_limit])
        else:
            return arr

    # Case 1 (patten == base): only one check must be performed
    # More formally: dim(pattern) == dim(base)
    if all(pattern_shape[i] == base_shape[i] for i in range(2)):
        return np.array([1])

    # Case 2 (pattern < base): The pattern must shrink AND grow
    # More formally: nrows(pattern) < nrows(base) AND ncols(pattern) < ncols(base)
    elif all(pattern_shape[i] < base_shape[i] for i in range(2)):
        upper_bound = np.min(np.array(base_shape) / np.array(pattern_shape))
        lower_bound = prop_shrink
        step = (upper_bound - lower_bound) * prop_shrink
        bounds_array = np.arange(lower_bound, upper_bound, step)

    # Case 3 (pattern > base): The pattern must only shrink
    # More formally: nrows(pattern) > nrows(base) OR ncols(pattern) > ncols(base)
    elif any(pattern_shape[i] > base_shape[i] for i in range(2)):
        upper_bound = 1 / np.max(np.array(pattern_shape) / np.array(base_shape))
        lower_bound = upper_bound * prop_shrink
        bounds_array = np.linspace(upper_bound, lower_bound, prop_shrink * 100)

    # Impose lower limit and add idenity transformation.
    return np.append(bounds_clean(bounds_array), 1)


def _best_guess_location(match_template_result):
    """

    Takes the result of skimage.feature.match_template() and returns (top left x, top left y)

    :param match_template_result:
    :return:
    """
    ij = np.unravel_index(np.argmax(match_template_result), match_template_result.shape)
    x, y = ij[::-1]
    return x, y


def _scale_invar_match_template(pattern_img, base_img, base_top_cropping=1/3.0):
    """

    Algorithm:
      1. Loop through various scales of the pattern image.
          2. Resize the pattern image and convert it grayscale
          3. Crop to the image to `base_top_cropping`.
          4. Run skimage.feature.match_template algorithm.
          5. Get top left and bottom right of the bounding box.
      5. Select the best match.
      6. Compute the full bounding box.
      7. Return the best match.

    :param pattern_img:
    :param base_img:
    :param base_top_cropping: crops the base image to the top x proportion.
    :return:
    """
    def corners_calc(top_left, bottom_right):
        return {
            'top_left': top_left,
            'top_right': (top_left[0] + bottom_right[0] - top_left[0], top_left[1]),
            'bottom_left': (top_left[0], top_left[1] + bottom_right[1]),
            'bottom_right': bottom_right}

    d = dict()
    for pattern_scale in _bounds_sinvar_t_match(pattern_img.shape[0:2], base_img.shape[0:2], prop_shrink=0.075):
        pattern_img_scaled = rgb2gray(imresize(pattern_img, pattern_scale, interp='lanczos'))

        # Crop
        top_third = int(base_img.shape[0] * base_top_cropping)
        base_img_cropped = base_img[:top_third]

        # Run the match_template algorithm
        result = match_template(base_img_cropped, pattern_img_scaled) # often just a sparse matrix.
        match_quality = result.max()

        # Top Left Corner of the bounding box
        top_left = np.array(best_guess_location(result))

        # Note the need for pattern_img_scaled.shape to go from (HEIGHT, WIDTH) to (WIDTH, HEIGHT).
        bottom_right = top_left + np.array(pattern_img_scaled.shape)[::-1]

        d[pattern_scale] = (match_quality, (tuple(top_left), tuple(bottom_right)))

    best_match = max(d, key=lambda x: d.get(x)[0])
    return {"match_quality": d[best_match][0],
            "box": corners_calc(top_left=d[best_match][1][0], bottom_right=d[best_match][1][1])}


def robust_match_template(pattern_img_path, base_img_path):
    """

    Search for a pattern image in a base image using a algorithm which is robust
    against variation in the size of the pattern in the base image.

    :param pattern_img_path:
    :param base_img_path:
    :return:
    """
    pattern_img = imread(pattern_img_path)
    base_img = imread(base_img_path, flatten=True)

    return robust_match_template(pattern_img, base_img)


def _robust_match_template_plot(d, best_scale, pattern_img, base_img):
    """

    :param d:
    :param best_scale:
    :param pattern_img: pattern_img (note, do not apply grayscale transformation).
    :param base_img:
    :return:
    """
    import matplotlib.pyplot as plt
    w, h = np.abs(np.array(d[best_scale][1][0]) - np.array(d[best_scale][1][1]))

    # Plot
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, adjustable='box-forced')

    ax1.imshow(rgb2gray(pattern_img))
    ax1.set_axis_off()
    ax1.set_title('pattern_img')

    ax2.imshow(base_img)
    ax2.set_axis_off()
    ax2.set_title('base_img')
    rect = plt.Rectangle(box_top_left, w, h, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    plt.show()





# ------------------------------------------------------------------------------------------
# Image Classification
# ------------------------------------------------------------------------------------------


# Load the CNN
ircnn = ImageRecognitionCNN(data_path)
ircnn.conv_net()
ircnn.load(os.path.join(data_path, "10_epoch_new_model_2.h5"), override_existing=True)

# Reverse the classes dict
reversed_data_classes = dict_reverse(ircnn.data_classes)


def class_guess(name, img_path, data_classes, img_shape):
    """

    :param img_path:
    :param data_classes:
    :return:
    """
    if name is None or pd.isnull(name):
        return np.NaN

    # Load the image and rescale
    img = load_img(os.path.join(img_path, name), target_size=img_shape).convert('RGB')

    # Convert to numpy array
    x = np.asarray(img, dtype='float32')
    x = x.transpose(2, 0, 1)
    x = np.expand_dims(x, axis=0)

    # ToDo: pass `ircnn.model` in when this is converted to a class.
    prediction = ircnn.model.predict_classes(x, verbose=0)[0]
    class_guess = data_classes[prediction]

    return class_guess

# df = df.drop_duplicates('img_cache_name')
#
# # Classify the images
# df['image_classification'] = df['img_cache_name'].progress_map(
#     lambda x: class_guess(x, raw_img_path, reversed_data_classes, ircnn.img_shape)
# )
#
#
# a = df[df['image_classification'].str.contains('grid')]['img_cache_name'].tolist()
# len(a)
# a
#
# from random import shuffle
# from time import sleep
# shuffle(a)
#
#
# for i in a[0:20]:
#     img_text(os.path.join(raw_img_path, i))
#     sleep(2)
#
#







































































