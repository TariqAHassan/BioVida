"""

    Border and Line Detection Algorithms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import numpy as np
import pandas as pd
from itertools import groupby
from skimage.feature import canny
# from matplotlib import pyplot as plt
from scipy.misc import imread, imshow
from skimage.color.colorconv import rgb2gray

# Note: top crop vertically: img[:, 0:200] (200 = end point)
# Want: the three largest jutaposed chages in absolute value.

# ToDo: use var instead of std to avoid an expensive sqrt operation.


# img = rgb2gray(imread(p, flatten=True)) / 255


# show_plt(border_removal(img))
# show_plt(img)


def show_plt(image):
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


def img_crop(img, h_crop, v_crop):
    # Crop above and below the hlines
    h_crop = img[horizontal_lines[0]:horizontal_lines[1]]
    # Crop to the left and right of the vertical lines
    full_crop = h_crop[:, vertical_lines[0]:vertical_lines[1]]

    return full_crop


def rolling_avg(iterable, window=8):
    """

    :param iterable:
    :param window:
    :return:
    """
    return pd.Series(iterable).rolling(center=False,window=window).mean().dropna()


def deltas(iterable):
    """

    Compute the deltas between all adjacent elements in ``iterable``.

    :param iterable:
    :return:
    """
    # Source: http://stackoverflow.com/a/2400875/4898004.
    return np.array([abs(j-i) for i, j in zip(iterable, iterable[1:])])


def largest_n_values(arr, n):
    """

    Returns the index (i.e., rows or columns) of the largest n numbers

    :param arr:
    :type arr: ``1D ndarray``
    :param n:
    :return: sorted list of the ``n`` largest values in ``arr`` (ascending order).
    """
    return tuple(sorted(arr.argsort()[-n:]))


def zero_blocks(axis_deltas, block_trehsold, count_threshold, round_to):
    """

    Tool which checks for number of blocks of color with litte variation.

    :param axis_deltas: *MUST* be the output of the deltas() method.
    :param block_trehsold: number of zeros in a row to be considered a block.
    :param count_threshold: number of blocks required for the function to return True.
    :param round_to: round
    :return:
    """
    axis_deltas_rounded = np.round(axis_deltas, round_to)

    # Get the number of sections with rows/columns with no difference
    sections = [len(list(j)) for i, j in groupby(axis_deltas_rounded) if i == 0]

    # Compute the number of zero var delta sections which are >= than the
    # number required for it to be considered a 'zero section' (e.g., two
    # rows with zero delta shouldn't be considered a 'zero section').
    n_zero_blocks = sum(x >= block_trehsold for x in sections)

    # Check if sufficent blocks exist.
    return n_zero_blocks >= count_threshold


def frame_detection(hdeltas, vdeltas, prop_img_solid_col_for_border=0.05, count_threshold=2, round_to=3):
    """


    :param hdeltas:
    :param vdeltas:
    :param prop_img_solid_col_for_border: proportion of colomns/rows required to have
                                low variance (i.e., be a color block).
    :param count_threshold:
    :param round_to:
    :return:
    """
    def fd_func(x):
        block_trehsold = int(len(x) * prop_img_solid_col_for_border)
        return zero_blocks(x, block_trehsold, count_threshold, round_to)
    return fd_func(hdeltas), fd_func(vdeltas)


def var_calc(img):
    """

    :param img:
    :return:
    """
    # Compute the variance for each row of the image
    horizontal_var = np.std(img, axis=1)
    # Compute the variance for each column of the image
    vertical_var = np.std(img, axis=0)
    return horizontal_var, vertical_var


def border_removal(horizontal_var, vertical_var, rolling_window):
    """

    :param horizontal_var:
    :param vertical_var:
    :param rolling_window:
    :return:
    """
    # Clean var arrays with rolling average; Recompute the deltas on the cleaned arrays
    horizontal_lines = None
    vertical_lines = None

    if not isinstance(horizontal_var, bool):
        # Note: n = 3 for h lines to account for the possibler lower band in MedPix images
        horizontal_var_cln = rolling_avg(horizontal_var, rolling_window).as_matrix()
        hdeltas = deltas(horizontal_var)
        # Get the lines based on which are largest
        horizontal_lines = largest_n_values(hdeltas, n=3)[:2]

    if not isinstance(vertical_var, bool):
        vertical_var_cln = rolling_avg(vertical_var, rolling_window).as_matrix()
        vdeltas = deltas(vertical_var)
        vertical_lines = largest_n_values(vdeltas, n=2)

    return horizontal_lines, vertical_lines


def row_most_edges(img, hbar_criteria):
    """

    :param img:
    :param hbar_criteria:
    :return:
    """
    bottom_proprtion_img = hbar_criteria['lower_prop']
    threshold = hbar_criteria['threshold']
    canny_sigma= hbar_criteria['canny_sigma']

    crop_location = int(img.shape[0] * 0.75)
    cropped_img = img[crop_location:]

    # Find the edges
    edges = canny(cropped_img, sigma=canny_sigma)

    # Find row with the most 'edges' (suggesting a line)
    row_with_the_most_edges = max(enumerate(edges), key=lambda x: sum(x[1]))

    # Check that proportion of edges in that row is greater than some threshold.
    if sum(row_with_the_most_edges[1]) < edges.shape[1]*threshold:
        return None

    # Return the edge line
    return crop_location + row_with_the_most_edges[0]


def line_analysis(img
                  , frame_detect_criteria={"prop_img_solid_col_for_border": 0.05, "count_threshold": 2, "round_to": 3}
                  , hbar_criteria={"lower_prop": 0.75, "threshold": 0.85, "canny_sigma": 0.25}
                  , rollow_window_for_border_removal=5
                  , check_lower_band=True):
    """

    :param img:
    :param frame_detect_criteria:
    :type frame_detect_criteria: ``dict``
    :param hbar_criteria: dict of the form:
                                    {'lower_prop' (lower proportion of the image to search): ...,
                                     'threshold' (proportion of columns in the image that must have 'edges'): ...,
                                     'canny_sigma' (sigma value to pass to the canny algo.): ...}
    :type hbar_criteria: ``dict``
    :param check_lower_band:
    :param rollow_window_for_border_removal:
    :type rollow_window_for_border_removal: ``int``
    :return:
    :rtype: ``dict``
    """
    # Compute the variances for the horizontal and vertical axes
    horizontal_var, vertical_var = var_calc(img)
    
    # Compute the deltas (between each adjacent row/column)
    hdeltas = deltas(horizontal_var)
    vdeltas = deltas(vertical_var)
    
    # Check if there is a horizontal border, vertical border
    frame_existance = frame_detection(hdeltas, vdeltas,
                                      prop_img_solid_col_for_border=frame_detect_criteria['prop_img_solid_col_for_border'],
                                      count_threshold=frame_detect_criteria['count_threshold'],
                                      round_to=frame_detect_criteria['round_to'])
    h_exists, v_exists = frame_existance

    # Lines dictionary
    d = dict.fromkeys(['hlines', 'vlines', 'lower_band'], None)

    if not any(frame_existance):
        if check_lower_band:
            d['lower_band'] = row_most_edges(img, hbar_criteria)
    elif any(frame_existance):
        h_lines, v_lines = border_removal(horizontal_var=horizontal_var if h_exists else h_exists,
                                          vertical_var=vertical_var if v_exists else v_exists,
                                          rolling_window=rollow_window_for_border_removal)

        if check_lower_band:
            d['lower_band'] = row_most_edges(img, hbar_criteria)
        if h_exists:
            d['hlines'] = h_lines
        if v_exists:
            d['vlines'] = v_lines

    return d


# line_analysis(img)
























