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

from keras.preprocessing.image import (ImageDataGenerator,
                                       array_to_img,
                                       img_to_array,
                                       load_img)

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
#   2. Look for MedLine(R) logo
#         - if true, crop
#   3. Look for text bar *
#         - if true, try crop, else ban
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
#
# Notes:
#     border removal: http://stackoverflow.com/q/10615901/4898004
#
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
# df = opi.pull()

# Save or Load the database.
# opi.cache('image_record_database', action='save')
# df = opi.current_search_dataframe
# df = opi.image_record_database

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
    :rtype: ``bool``
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
# Run OCR on the images
# ------------------------------------------------------------------------------------------

# img = Image.open(image_path)
# pytesseract.image_to_string(img)
# print(pytesseract.image_to_string(img, boxes=True))
#
#
# print(pytesseract.image_to_string(img))

# ------------------------------------------------------------------------------------------
# Watermark Cleaning
# ------------------------------------------------------------------------------------------

from biovida.images.models.template_matching import robust_match_template
from biovida.images.models.temp import resources_path

base_images = [os.path.join(raw_img_path, i) for i in os.listdir(raw_img_path) if i.endswith(".png")]

def logo_analysis(x, threshold=0.25, x_greater_check=1/3.0, y_greater_check=1/2.0):
    """

    Wraps ``biovida.images.models.template_matching.robust_match_template()``.

    :param x: passed from df.apply().
    :type x: ``Pandas Series``
    :param threshold:
    :type threshold: ``float``
    :param x_greater_check:
    :type x_greater_check: ``float``
    :param y_greater_check:
    :type y_greater_check: ``float``
    :return:
    :rtype: ``tuple``
    """
    # if not grayscale, skip.

    if 'medpix' not in x['journal_title'].lower():
        return np.NaN

    # Pattern
    medline_template_img = os.path.join(resources_path, "medpix_logo.png")

    # The image to check
    base_images_p = x['img_cache_name_full']

    # Run the matching algorithm
    match, base_img_shape = robust_match_template(pattern_img_path=medline_template_img, base_img_path=base_images[0])

    # Check match quality
    if match['match_quality'] < threshold:
        return np.NaN

    # Check the box is in the top right
    box_bottom_left = match['box']['bottom_left']

    if box_bottom_left[0] < (base_img_shape[0] * x_greater_check) or \
        box_bottom_left[1] > (base_img_shape[1] * y_greater_check):
        return np.NaN
    else:
        return box_bottom_left


df['logo_loc'] = df.progress_apply(logo_analysis, axis=1)


# ------------------------------------------------------------------------------------------
# Image Classification
# ------------------------------------------------------------------------------------------


# Load the CNN
ircnn = ImageRecognitionCNN(data_path)
ircnn.load(os.path.join(data_path, "10_epoch_new_model_2.h5"), override_existing=True, default_model_load=True)

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







































































