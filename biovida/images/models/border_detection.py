"""

    Border and Edge Detection Algorithms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import os
import numpy as np
import pandas as pd
from operator import sub
from copy import deepcopy
from functools import reduce
from scipy.misc import imread, imshow
from skimage.color.colorconv import rgb2gray
from matplotlib import pyplot as plt


def _load_img_rescale(path_to_image):
    """

    :param path_to_image:
    :return:
    """
    return rgb2gray(imread(path_to_image, flatten=True)) / 255.0


def _show_plt(image):
    """

    :param image:
    :return:
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


def rounder(l, by=3):
    """

    :param l:
    :param by:
    :return:
    """
    t = int if by == 0 else float
    return list(map(lambda x: round(t(x), by), l))


def min_max(l):
    return [min(l), max(l)]


def column_round(l):
    """

    :param l:
    :return:
    """
    return rounder(np.median(l, axis=0), 0)


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


def subsection(numeric_array, exclude, start, end):
    """

    :param numeric_array:
    :param exclude:
    :param start:
    :param end:
    :return:
    """
    return numeric_array[start:exclude] + numeric_array[exclude+1:end]


def anomally_removal(numeric_array, window=4):
    """

    Replaces single numbers that do not match their homogenous neighbourhood
    with the value of the homogenious neighbood.

    E.g., 0, 0, 0, 0, 99, 0, 0, 0, 0 --> 0, 0, 0, 0, 0, 0, 0, 0, 0.

    :param numeric_array: must be a list or tuple. Numpy array will break this.
    :param window:
    :return:

    Example:
    -------
    >>> row = [0.24677, 0.24677, 0.24677, 0.9, 0.24677, 0.24677, 0.24677]
    >>> for i, j in zip(row, anomally_removal(row, 3)):
    ...     print(i, "-->", j, ".Change Made: ", i != j)
        0.24677 --> 0.24677 .Change Made:  False
        0.24677 --> 0.24677 .Change Made:  False
        0.24677 --> 0.24677 .Change Made:  False
        0.9     --> 0.24677 .Change Made:  True
        0.24677 --> 0.24677 .Change Made:  False
        0.24677 --> 0.24677 .Change Made:  False
        0.24677 --> 0.24677 .Change Made:  False
    """
    # Note: the logic is somewhat complex here...a mistake is possible.

    if window <= 1 or window > len(numeric_array):
        raise ValueError("`window` must be greater than 1 and less than the length of `numeric_array`.")

    if window > 2:
        window += 1

    smoothed = list()
    for i in range(len(numeric_array)):
        # if can't look back
        if i < window:
            # If all forward neighbours are the same
            if len(set(numeric_array[i+1:i+window+1])) == 1:
                # Smooth using the next element
                smoothed.append(numeric_array[i+1])
            else:
                # if not, leave 'as is'
                smoothed.append(numeric_array[i])
        # if can look back
        if i >= window:
            if len(set(subsection(numeric_array, i, i-window+1, i+window))) == 1:
                # smooth using the prior element (needed to prevent error at the end of the iterable).
                smoothed.append(numeric_array[i-1])
            else:
                smoothed.append(numeric_array[i])

    return smoothed


def rolling_avg(iterable, window):
    """

    :param iterable:
    :param window:
    :return:
    """
    return pd.Series(iterable).rolling(center=False, window=window).mean().dropna()


def array_cleaner(arr, round_by=5, rolling_window=5, anomally_window=2):
    """

    :param arr:
    :param round_by:
    :return:
    """
    rolling = rolling_avg(arr, rolling_window)
    rounded_arr = rounder(rolling, round_by)
    return np.array(anomally_removal(rounded_arr, anomally_window))


def largest_n_changes_with_values(iterable, n):
    """Compute the index of the ``n`` largest deltas
       the their associated values.
    """
    large_ds = largest_n_values(deltas(iterable), n)

    # Look around for true smallest value for each.
    # i.e., index of the value which triggered the low delta.
    true_smallest = list()
    for d in large_ds:
        if d == 0 or len(iterable) == 2:
            neighbours = [0, 1]
        elif d == len(iterable):
            neighbours = [-1, 0]
        else:
            neighbours = [-1, 0, 1]
        options = [(d+i, iterable[d+i]) for i in neighbours]
        flr = [i for i in options if i[1] == min((j[1] for j in options))][0][0]
        true_smallest.append(flr)

    return [(d, iterable[d]) for d in true_smallest]


def expectancy_violation(expected, actual):
    """

    :param expected:
    :param actual:
    :return:
    """
    val = float(abs(expected - actual) / expected)
    return round(val, 4)


def largest_median_inflection(averaged_axis_values, axis):
    """

    :param averaged_axis_values:
    :param axis: 0 = column; 1 = rows
    :return:
    """
    if axis == 1:
        n_largest = 4
    elif axis == 0:
        n_largest = 2

    # Postion of largest changes
    large_inflections = largest_n_changes_with_values(averaged_axis_values, n_largest)

    # Sort by position
    large_inflections_sorted = sorted(large_inflections, key=lambda x: x[0])

    # Compute the media for the whole axis
    median = np.median(averaged_axis_values)

    # Compute the how much the signal deviated from the median value (0-1).
    median_deltas = [(i, expectancy_violation(median, j)) for (i, j) in large_inflections_sorted]

    if axis == 1:
        return median_deltas[1:]
    elif axis == 0:
        return median_deltas


def zero_var_axis_elements_remove(img, axis, rounding=3):
    """

    Replaces, by axis, matrix elements with approx. no variance
    (technically using the standard deviation here).

    :param img:
    :param axis:
    :param rounding:
    :return:
    """
    zero_var_items = np.where(np.round(np.std(img, axis=axis), rounding) == 0)
    if axis == 0:
        img[:, zero_var_items] = [0]
    elif axis == 1:
        img[zero_var_items] = [0]
    return img


def edge_dection(img, axis=0):
    """

    :param img:
    :param axis: 0 = column; 1 = rows
    :return:
    """
    # Deep copy the matrix to prevent side effects.
    # img = image_array.copy()

    # Set rows with no ~variance to zero vectors to eliminate their muffling effect on the signal.
    img = zero_var_axis_elements_remove(img, axis)

    # Average the remaining values
    averaged_axis_values = np.mean(np.abs(img), axis=axis)

    return largest_median_inflection(averaged_axis_values, axis)


def evidence_weigh(candidates, axis_size, signal_strength_threshold, min_border_seperation):
    """

    :param candidates:
    :param axis_size:
    :param signal_strength_threshold:
    :param min_border_seperation:
    :return:
    """
    conclusion = None
    if all(x[1] >= signal_strength_threshold for x in candidates):
        if abs(reduce(sub, [i[0] for i in candidates])) >= (min_border_seperation * axis_size):
            conclusion = candidates

    return conclusion


def border_detection(image_arr
                     , signal_strength_threshold=0.25
                     , min_border_seperation=0.15
                     , lower_bar_search_space=0.90
                     , report_evidence=False):
    """

    :param image_array:
    :param signal_strength_threshold:
    :param min_border_seperation:
    :param lower_bar_search_space:
    :param report_evidence:
    :return:
    """
    image_array = deepcopy(image_arr)

    # Initalize the return dict
    d = dict.fromkeys(['vborder', 'hborder', 'hbar'], None)

    # Get Values for columns
    v_edge_candidates = edge_dection(image_array, axis=0)
    # Run Analysis
    d['vborder'] = evidence_weigh(v_edge_candidates, image_array.shape[1], signal_strength_threshold, min_border_seperation)

    # Get Values for rows
    h_border_candidates = edge_dection(image_array, axis=1)

    # Run Analysis (exclude final element in `h_border_candidates` -- analyze below).
    d['hborder'] = evidence_weigh(h_border_candidates[:2], image_array.shape[1], signal_strength_threshold, min_border_seperation)

    # Checks for the lower bar:
    #   1. That its signal is greater than or equal to `signal_strength_threshold`.
    #   2. That it lies at or below `lower_bar_search_space`.
    lower_bar_candidate = h_border_candidates[-1]
    if lower_bar_candidate[1] > signal_strength_threshold:
        if (image_array.shape[0] * lower_bar_search_space) <= lower_bar_candidate[0]:
            d['hbar'] = lower_bar_candidate

    # Return the analysis
    if report_evidence:
        return d
    else:
        return {k: [i[0] for i in v] if isinstance(v, list) else (v if v is None else v[0]) for k, v in d.items()}


def _lines_plotter(path_to_image):
    """

    Visualizes the default line_analysis settings

    :param lines_dict:
    :return:
    """
    from matplotlib import pyplot as plt
    from matplotlib import collections as mc

    image = _load_img_rescale(path_to_image)
    analysis = {k: v for k, v in border_detection(image_arr=image).items() if v is not None}

    h, w = image.shape
    h_lines_explicit = [[(0, i), (w, i)] for i in analysis.get('hborder', [])]
    v_line_explicit = [[(i, 0), (i, h)] for i in analysis.get('vborder', []) if i is not None]
    lines = h_lines_explicit + v_line_explicit

    if "int" in str(type(analysis.get('hbar', None))):
        lines += [[(0, int(analysis['hbar'])), (w, int(analysis['hbar']))]]

    if len(lines):
        line_c = mc.LineCollection(lines, colors=['r'] * len(lines), linewidths=2)
        fig, ax = plt.subplots()
        ax.add_collection(line_c)
        ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        plt.show()
        print(analysis)
        return True
    else:
        fig, ax = plt.subplots()
        ax.text(0, 0, "No Results to Display", fontsize=15)
        ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        # print("No Results to Display.")
        print(analysis)
        return False


















































































































































