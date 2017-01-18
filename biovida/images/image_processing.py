"""

    Image Processing
    ~~~~~~~~~~~~~~~~

"""
# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageStat
from tqdm import tqdm
from pprint import pprint

# Tools form the image subpackage
from biovida.images.resources import * # needed to generate medpix logo. Refactor.
from biovida.images.models.temp import resources_path
from biovida.images.openi_interface import OpenInterface

# Models
from biovida.images.models.border_detection import border_detection
from biovida.images.models.img_classification import ImageRecognitionCNN
from biovida.images.models.template_matching import robust_match_template

# Tool to create required caches
from biovida.support_tools._cache_management import _package_cache_creator

# Start tqdm
tqdm.pandas("status")

# Suppress np scientific notation
np.set_printoptions(suppress=True)

# ---------------------------------------------------------------------------------------------
# General procedure (for Ultrasound, X-ray, CT and MRI):
# ---------------------------------------------------------------------------------------------
#
#   1. Check if grayscale                                                       X
#         - if false, try to detect a frame
#               - if frame, crop else ban
#   2. Look for MedLine(R) logo                                                 P
#         - if true, crop
#   3. Look for text bar                                                        P
#         - if true, try crop, else ban
#   4. Look for border                                                          P
#         - if true, try crop, else ban
#   5. Look for arrows or boxes *
#         - if true, ban
#   6. Look for image grids *
#         - if true, ban
#   7. Look for graphs *
#         - if true, ban
#   8. Look for faces *
#         - if true, ban
#   9. Look for other text in the image *
#         - if true, ban (or find crop).
#
# Legend:
#     * = requires machine learning
#     p = partially solved
#     X = solved
#
# ---------------------------------------------------------------------------------------------
# Temporary Data
# ---------------------------------------------------------------------------------------------

from biovida.images.models.temp import cache_path

# Create an instance of the Open-i data harvesting tool
opi = OpenInterface(cache_path, download_limit=5500, img_sleep_time=2, records_sleep_main=None)

# Get the Data
opi.search(None, image_type=['mri'])

df = opi.image_record_database

# ---------------------------------------------------------------------------------------------
# Grayscale Analysis
# ---------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------
# Watermark Cleaning
# ---------------------------------------------------------------------------------------------

def logo_analysis(x, threshold=0.25, x_greater_check=1/3.0, y_greater_check=1/2.5):
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
    # ToDo: if not grayscale, skip.

    if 'medpix' not in x['journal_title'].lower():
        return np.NaN

    # Pattern
    medline_template_img = os.path.join(resources_path, "medpix_logo.png")

    # The image to check
    base_images_p = x['img_cache_name_full']

    # Run the matching algorithm
    match, base_img_shape = robust_match_template(pattern_img_path=medline_template_img, base_img_path=base_images_p)

    # Check match quality
    if not isinstance(match, dict) or match['match_quality'] < threshold:
        return np.NaN

    box_bottom_left = match['box']['bottom_left']

    # Check the box is in the top right
    if box_bottom_left[0] < (base_img_shape[0] * x_greater_check) or \
        box_bottom_left[1] > (base_img_shape[1] * y_greater_check):
        return np.NaN
    else:
        return box_bottom_left

def _logo_plotting(found_logo_df, start=0, end=10):
    """
    found_logo_df = df[pd.notnull(df['logo_loc'])].copy()
    :param found_logo_df:
    :return:
    """
    import matplotlib.pyplot as plt
    from scipy.misc import imread
    from time import sleep

    def _logo_diving_plotter(base_img, dividing_line):
        fig = plt.figure(figsize=(10, 6))
        ax1 = plt.subplot(1, 1, 1)
        ax1.imshow(base_img, cmap='gray')
        ax1.set_axis_off()
        ax1.axhline(y=dividing_line, color='r', linestyle='-')
        ax1.set_title('Diving Choice')
        plt.show()

    img_dividing_lines = pd.Series(found_logo_df['logo_loc'].values, index=found_logo_df['img_cache_name_full']).to_dict()
    for (k, v) in list(img_dividing_lines.items())[start:end]:
        _logo_diving_plotter(imread(k, flatten=True), v[1])
        sleep(1)

# ---------------------------------------------------------------------------------------------
# Detect Border
# ---------------------------------------------------------------------------------------------

def border_analysis(data_frame
                     , join=True
                     , signal_strength_threshold=0.25
                     , min_border_separation=0.15
                     , lower_bar_search_space=0.9
                     , report_signal_strength=False):
    """

    Wrapper for ``biovida.images.models.border_detection.border_detection()``.

    :param data_frame:
    :param join: if ``True``, left join with ``data_frame``. Defaults to True.
    :param signal_strength_threshold:
    :param min_border_separation:
    :param lower_bar_search_space:
    :param report_signal_strength:
    :return:
    """
    def ba_func(image):
        return border_detection(image,
                                signal_strength_threshold,
                                min_border_separation,
                                lower_bar_search_space,
                                report_signal_strength)

    # Run the analysis
    border_analysis = data_frame['img_cache_name_full'].progress_map(ba_func)

    # Convert to a dataframe and return
    ba_df = pd.DataFrame(border_analysis.tolist()).fillna(np.NaN)
    return data_frame.join(ba_df, how='left') if join else ba_df

borders_and_edges = border_analysis(df, join=True)

# ---------------------------------------------------------------------------------------------
# Image Classification
# ---------------------------------------------------------------------------------------------

from biovida.images.models.temp import cnn_model_location

# Load the CNN
ircnn = ImageRecognitionCNN()
ircnn.load(cnn_model_location, override_existing=True)

# Get classes
# m = ircnn.predict([])
# pd.Series(m)




















































