"""

    Border and Edge Detection Algorithms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import numpy as np
import pandas as pd
from operator import sub
from copy import deepcopy
from functools import reduce
from scipy.misc import imread, imshow
from skimage.color.colorconv import rgb2gray


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


def anomaly_removal(numeric_array, window=2):
    """

    Replaces single numbers that do not match their homogeneous neighbourhood
    with the value of the homogeneous neighbors.

    E.g., 0, 0, 0, 0, 99, 0, 0, 0, 0 --> 0, 0, 0, 0, 0, 0, 0, 0, 0.

    :param numeric_array: must be a list or tuple. Numpy array will break this.
    :param window:
    :return:

    Example:
    -------
    >>> row = [0.24677, 0.24677, 0.24677, 0.9, 0.24677, 0.24677, 0.24677]
    >>> for i, j in zip(row, anomaly_removal(row, 3)):
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


def array_cleaner(arr, round_by=5, anomaly_window=2):
    """

    :param arr:
    :param round_by:
    :param rolling_window:
    :param anomaly_window:
    :return:
    """
    rounded_arr = rounder(arr, round_by)
    return np.array(anomaly_removal(rounded_arr, anomaly_window))


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
    return round(float(abs(expected - actual) / expected), 4)


def largest_median_inflection(averaged_axis_values, axis, n_largest_override=None):
    """

    :param averaged_axis_values:
    :param axis: 0 = column; 1 = rows
    :return:
    """
    if isinstance(n_largest_override, int):
        n_largest = n_largest_override
    elif axis == 1:
        n_largest = 4
    else:
        n_largest = 2

    # Position of largest changes
    large_inflections = largest_n_changes_with_values(averaged_axis_values, n_largest)

    # Sort by position
    large_inflections_sorted = sorted(large_inflections, key=lambda x: x[0])

    # Compute the media for the whole axis
    median = np.median(averaged_axis_values)

    # Compute the how much the signal deviated from the median value (0-1).
    median_deltas = [(i, expectancy_violation(median, j)) for (i, j) in large_inflections_sorted]

    if isinstance(n_largest_override, int):  # neighborhood search in largest_n_changes_with_values may --> duplicates.
        return median_deltas
    elif axis == 1:
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


def edge_detection(img, axis=0, n_largest_override=None):
    """

    :param img:
    :param axis: 0 = column; 1 = rows
    :param n_largest_override:
    :return:
    """
    # Set rows with no ~variance to zero vectors to eliminate their muffling effect on the signal.
    img = zero_var_axis_elements_remove(img, axis)

    # Average the remaining values
    # ToDo: it's not not clear if array_cleaner() helps much...after all, the vector has already been averaged.
    averaged_axis_values = array_cleaner(np.mean(img, axis=axis))

    return largest_median_inflection(averaged_axis_values, axis, n_largest_override)


def evidence_weigh(candidates, axis_size, signal_strength_threshold, min_border_separation ):
    """

    Weight the evidence that a true border has been detected.

    :param candidates:
    :param axis_size:
    :param signal_strength_threshold:
    :param min_border_separation :
    :return:
    """
    midpoint = (axis_size / 2)
    left_buffer = midpoint - (axis_size * 1/15)
    right_buffer = midpoint + (axis_size * 1/15)

    conclusion = None
    if all(x[1] >= signal_strength_threshold for x in candidates):
        if abs(reduce(sub, [i[0] for i in candidates])) >= (min_border_separation  * axis_size):
            if candidates[0][0] < left_buffer and candidates[1][0] > right_buffer:
                conclusion = candidates

    return conclusion


def lower_bar_detection(image_array, lower_bar_search_space, signal_strength_threshold, cfloor=None):
    """

    :param image_array:
    :param lower_bar_search_space:
    :param signal_strength_threshold:
    :param cfloor: check floor. If None, the floor is the last image in the photo.
    :return:
    """
    # Compute the location to crop the image
    cut_off = int(image_array.shape[0] * lower_bar_search_space)
    flr = (image_array.shape[0] if not isinstance(cfloor, int) else cfloor)

    if abs(cut_off - flr) < 3:
        return cfloor if cfloor is not None else None

    lower_image_array = image_array.copy()[cut_off:flr]

    # Run an edge analysis
    lower_bar_candidates = edge_detection(img=lower_image_array, axis=1, n_largest_override=8)

    # Apply Threshold
    thresholded_values = list(set([i[0] + cut_off for i in lower_bar_candidates if i[1] > signal_strength_threshold]))

    if not len(thresholded_values):
        return None

    m = np.mean(thresholded_values).astype(int)
    # Return the averaged guess
    return int(m)


def double_pass_lower_bar_detection(image_array, lower_bar_search_space, signal_strength_threshold):
    """

    :param image_array:
    :param lower_bar_search_space:
    :param signal_strength_threshold:
    :return:
    """
    first_pass = lower_bar_detection(image_array, lower_bar_search_space, signal_strength_threshold)
    second_pass = lower_bar_detection(image_array, lower_bar_search_space, signal_strength_threshold, cfloor=first_pass)

    return first_pass if second_pass is None else second_pass


def border_detection(image_arr
                     , signal_strength_threshold=0.25
                     , min_border_separation=0.15
                     , lower_bar_search_space=0.9
                     , report_signal_strength=False):
    """

    :param image_arr:
    :param signal_strength_threshold:
    :param min_border_separation :
    :param lower_bar_search_space: set to ``None`` to disable.
    :param report_signal_strength:
    :return:
    """
    image_array = deepcopy(image_arr)

    # Initalize the return dict
    d = dict.fromkeys(['vborder', 'hborder', 'hbar'], None)

    # Get Values for columns
    v_edge_candidates = edge_detection(image_array, axis=0)
    # Run Analysis
    d['vborder'] = evidence_weigh(v_edge_candidates, image_array.shape[1], signal_strength_threshold, min_border_separation )

    # Get Values for rows
    h_border_candidates = edge_detection(image_array, axis=1)

    # Run Analysis. This excludes final element in `h_border_candidates` as including a third elemnt
    # is simply meant to deflect the pull of the lower bar, if present.
    d['hborder'] = evidence_weigh(h_border_candidates[:2], image_array.shape[1], signal_strength_threshold, min_border_separation )

    # Look for lower bar
    d['hbar'] = double_pass_lower_bar_detection(image_array, lower_bar_search_space, signal_strength_threshold)

    # Return the analysis
    if report_signal_strength:
        return d
    else:
        return {k: [i[0] for i in v] if isinstance(v, list) else (v[0] if isinstance(v, (list, tuple)) else v) for k, v in d.items()}


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
    v_line_explicit = [[(i, 0), (i, h)] for i in analysis.get('vborder', [])]
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














































































































