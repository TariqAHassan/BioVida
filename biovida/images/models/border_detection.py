# coding: utf-8

"""

    Border and Edge Detection
    ~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np
import pandas as pd
from operator import sub
from functools import reduce


from biovida.images._image_tools import load_image_rescale


def _rounder(l, by=3):
    """

    Rounds all of the elements in an iterable.

    :param l: a list of values
    :type l: ``list`` or ``tuple``
    :param by:  If 0, the values will be converted to integers.
    :type by: ``int``
    :return: rounded numbers.
    :rtype: ``list``
    """
    t = int if by == 0 else float
    return list(map(lambda x: round(t(x), by), l))


def _deltas(iterable):
    """

    Compute the _deltas between all adjacent elements in ``iterable``.

    :param iterable: an iterable structure
    :type iterable: ``list`` or ``tuple``
    :return: an array with the delta between juxtaposed entries.
    :rtype: ``ndarray``
    """
    # Source: http://stackoverflow.com/a/2400875/4898004.
    return np.array([abs(j-i) for i, j in zip(iterable, iterable[1:])])


def _largest_n_values(arr, n):
    """

    Returns the index (i.e., rows or columns) of the largest n numbers

    :param arr: a numpy array
    :type arr: ``1D ndarray``
    :param n: the largest ``n`` values to return.
    :type n: ``int``
    :return: sorted list of the ``n`` largest values in ``arr`` (ascending order).
    :rtype: ``tuple``
    """
    return tuple(sorted(arr.argsort()[-n:]))


def _subsection(iterable, exclude, start, end):
    """

    Selects a subsection of an array, excluding a single subsection

    :param iterable: some iterable data structure, which can be indexed.
    :type iterable: ``list`` or ``tuple``
    :param exclude: the index to exclude.
    :type exclude: ``int``
    :param start: where to start the slice.
    :type start: ``int``
    :param end: where to end the slice
    :type end: ``int``
    :return: a list of the desired subsection.
    :rtype: ``list``
    """
    return iterable[start:exclude] + iterable[exclude+1:end]


def _anomaly_removal(numeric_array, window=2):
    """

    Replaces single numbers that do not match their homogeneous neighbourhood
    with the value of the homogeneous neighbors.

    E.g., 0, 0, 0, 0, 99, 0, 0, 0, 0 --> 0, 0, 0, 0, 0, 0, 0, 0, 0.

    :param numeric_array: must be a list or tuple. Numpy array will break this.
    :param window:
    :return: a list of smoothed values.
    :rtype: ``list``

    :Example:

    >>> row = [0.24677, 0.24677, 0.24677, 0.9, 0.24677, 0.24677, 0.24677]
    >>> for i, j in zip(row, _anomaly_removal(row, 3)):
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
            if len(set(_subsection(numeric_array, i, i-window+1, i+window))) == 1:
                # smooth using the prior element (needed to prevent error at the end of the iterable).
                smoothed.append(numeric_array[i-1])
            else:
                smoothed.append(numeric_array[i])

    return smoothed


def _rolling_avg(iterable, window):
    """

    Computes the rolling average using ``Pandas``.

    :param iterable: some iterable data structure, which can be indexed.
    :type iterable: ``list`` or ``tuple``
    :param window: the window for the rolling average.
    :type window: ``int``
    :return: the rolling average of the iterable. Note: center = ``False``.
    :rtype: ``Pandas Series``
    """
    return pd.Series(iterable).rolling(center=False, window=window).mean().dropna()


def _array_cleaner(arr, round_by=5, anomaly_window=2):
    """

    Round array elements and remove anomalies using ``_anomaly_removal()``.

    :param arr: a numpy array
    :type arr: ``ndarray``
    :param round_by: how much to round the input array
    :type round_by: ``int``
    :param anomaly_window: window on which to check for a replace anomalies. See _anomaly_removal().
    :type anomaly_window: ``int``
    :return: an array of the smoothed values.
    :rtype: ``ndarray``
    """
    rounded_arr = _rounder(arr, round_by)
    return np.array(_anomaly_removal(rounded_arr, anomaly_window))


def _largest_n_changes_with_values(iterable, n):
    """

    Compute the index of the ``n`` largest _deltas the their associated values.

    :param iterable: an iterable data structure which supports indexing.
    :type iterable: ``list`` or ``tuple``
    :param n: ``n`` largest.
    :type n: ``int``
    :return: a list of the index with largest changes, along with the values themselves
    :rtype: ``list``
    """
    large_ds = _largest_n_values(_deltas(iterable), n)

    # Look around for true smallest value for each.
    # i.e., the index of the value which triggered the large delta.
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


def _expectancy_violation(expected, actual, round_by=4):
    """

    Formula to compute the percent error.

    :param expected: the expected value for a given event.
    :type expected: ``float`` or ``int``
    :param actual: the actual value for a given event.
    :type actual: ``float`` or ``int``
    :param round_by: how much to round the result. Defaults to 4.
    :type round_by: ``int``
    :return: percent error
    :type: ``float``
    """
    # ToDo: explore better solutions.
    # This just blocks division by zero (given all values will be >= 0)
    expected += 0.0001
    actual += 0.0001

    val = round(float(abs(expected - actual) / expected), round_by)
    return round(float(val), round_by)


def _largest_median_inflection(averaged_axis_values, axis, n_largest_override=None):
    """

    Finds the values in a vector which diverge most strongly from the median value.

    :param averaged_axis_values: a 1D array which has been computed by averaging about a given axis in a matrix.
    :type averaged_axis_values: ``ndarray``
    :param axis: 0 for columns; 1 for rows.
    :type axis: ``int``
    :return: a list of rows/columns which diverge from the median,
             and the percent error as a metric of the strength of this divergence.
    :rtype: ``list``
    """
    if isinstance(n_largest_override, int):
        n_largest = n_largest_override
    elif axis == 1:
        n_largest = 4
    else:
        n_largest = 2

    # Position of largest changes
    large_inflections = _largest_n_changes_with_values(averaged_axis_values, n_largest)

    # Sort by position
    large_inflections_sorted = sorted(large_inflections, key=lambda x: x[0])

    # Compute the median for the whole axis
    median = np.median(averaged_axis_values)

    # Compute the how much the signal deviated from the median value (0-1).
    median_deltas = [(i, _expectancy_violation(median, j)) for (i, j) in large_inflections_sorted]

    if isinstance(n_largest_override, int):  # neighborhood search in _largest_n_changes_with_values may --> duplicates.
        return median_deltas
    elif axis == 1:
        return median_deltas[1:]
    elif axis == 0:
        return median_deltas


def _zero_var_axis_elements_remove(image, axis, rounding=3):
    """

    Replaces, by axis, matrix elements with approx. no variance
    (technically using the standard deviation here).

    :param image: an image represented as an array.
    :type image: ``ndarray``
    :param axis: 0 for columns; 1 for rows.
    :type axis: ``int``
    :param rounding: how much to round the standard deviation values for a given row/column.
    :type rounding: ``int``
    :return: a matrix with rows/columns with standard deviation == 0 replaced with zero vectors.
    :rtype: ``ndarray``
    """
    zero_var_items = np.where(np.round(np.std(image, axis=axis), rounding) == 0)
    if axis == 0:
        image[:, zero_var_items] = [0]
    elif axis == 1:
        image[zero_var_items] = [0]
    return image


def edge_detection(image, axis=0, n_largest_override=None):
    """

    Detects edges within an image.

    :param image: an image represented as a matrix
    :type image: ``ndarray``
    :param axis: 0 for columns; 1 for rows.
    :type axis: ``int``
    :param n_largest_override: override the defaults for the number of inflections to report
                               when searching the image along a given axis.
    :type n_largest_override: ``int`` or ``None``
    :return: the location of large inflections (changes) in the image along a given axis.
    :rtype: ``list``
    """
    # Set rows with no ~variance to zero vectors to eliminate their muffling effect on the signal.
    image = _zero_var_axis_elements_remove(image, axis)

    # Average the remaining values
    # ToDo: it's not not clear if ``_array_cleaner()`` helps much...after all, the vector has already been averaged.
    averaged_axis_values = _array_cleaner(np.mean(image, axis=axis))
    return _largest_median_inflection(averaged_axis_values, axis, n_largest_override)


def _weigh_evidence(candidates, axis_size, signal_strength_threshold, min_border_separation, buffer_multiplier=1/15):
    """

    Weigh the evidence that a true border has been detected.

    :param candidates: a list of lists or list of tuples
    :type candidates: ``list``
    :param axis_size: how long a given axis is, e.g., 512 (for a 512x256 image).
    :type axis_size: ``int``
    :param signal_strength_threshold: how strong the inflection must be relative
                                      to the median for the image. Must be between 0 and 1.
    :type signal_strength_threshold: ``float``
    :param min_border_separation: a value between 0 and 1 that determines the proportion of the axis
                                  that two edges must be separated for them to be considered borders.
                                  (i.e., ``axis_size`` * ``min_border_separation``)
    :type min_border_separation: ``float``
    :param buffer_multiplier: How far from the midpoint two lines must to be considered a border.
                              (calculation: midpoint +/- (axis_size * buffer_multiplier). Defaults to 1/15.
    :type buffer_multiplier: ``int`` or ``float
    :return: None if it was 'decided' that there was no enough evidence, else the candidates are returned 'as is'.
    :rtype: ``list`` or ``None``
    """
    midpoint = (axis_size / 2)
    lu_buffer = midpoint - (axis_size * buffer_multiplier)  # left/upper
    rl_buffer = midpoint + (axis_size * buffer_multiplier)  # right/lower

    conclusion = None
    if all(x[1] >= signal_strength_threshold for x in candidates):
        if abs(reduce(sub, [i[0] for i in candidates])) >= (min_border_separation * axis_size):
            if candidates[0][0] < lu_buffer and candidates[1][0] > rl_buffer:
                conclusion = candidates

    return conclusion


def _lower_bar_detection(image_array, lower_bar_search_space, signal_strength_threshold, cfloor=None):
    """

    Executes a single pass looking for a lower bar.

    :param image_array: an image represented as a numpy array.
    :type image_array: ``ndarray``
    :param lower_bar_search_space: a value between 0 and 1 specify the proportion of the image
                                   to search for a lower bar (e.g., 0.90).
    :type lower_bar_search_space: ``float``
    :param signal_strength_threshold: a value between 0 and 1 specify the signal strength required
                                      for an area required to be considered a 'lower bar'.
                                      Internally, this is measured as a location deviation from
                                      the median signal strength of the average image.
    :type signal_strength_threshold: ``int``
    :param cfloor: check floor. If None, the floor is the last image in the photo.
    :type cfloor: ``int`` or ``None
    :return: the height (y position) of the top of the lower bar (given one pass of this algorithm).
    :rtype: ``int``
    """
    # Compute the location to crop the image
    cut_off = int(image_array.shape[0] * lower_bar_search_space)
    flr = image_array.shape[0] if not isinstance(cfloor, int) else cfloor

    if abs(cut_off - flr) < 3:
        return cfloor if cfloor is not None else None  # redundant

    lower_image_array = image_array.copy()[cut_off:flr]

    lower_bar_candidates = edge_detection(image=lower_image_array, axis=1, n_largest_override=8)
    thresholded_values = list(set([i[0] + cut_off for i in lower_bar_candidates if i[1] > signal_strength_threshold]))

    if not len(thresholded_values):
        return None

    # Return the averaged guess
    mean_thresholded_values = np.mean(thresholded_values).astype(int)
    return int(mean_thresholded_values)


def lower_bar_analysis(image_array, lower_bar_search_space, signal_strength_threshold, lower_bar_second_pass):
    """

    Executes a two passes looking for a lower bar.
    This can be helpful if the lower border is striated in such a way that could confuse the
    algorithm. Namely, two the space between two lines of text can confuse the algorithm.
    The first pass with truncate the first line; the second pass will truncate the line above it.

    :param image_array: an image represented as a matrix.
    :type image_array: ``2D ndarray``
    :param lower_bar_search_space: a value between 0 and 1 specify the proportion of the image
                                   to search for a lower bar (e.g., 0.90).
    :type lower_bar_search_space: ``float``
    :param signal_strength_threshold: a value between 0 and 1 specify the signal strength required
                                      for an area required to be considered a 'lower bar'.
                                      This is measured as a absolute value of the difference between
                                      a location and the median signal strength of the average image.
    :type signal_strength_threshold: ``int`` or ``float``
    :param lower_bar_second_pass: if ``True`` perform a second pass on the lower bar.
    :type lower_bar_second_pass: ``bool``
    :return: the location of the start of the lower bar (i.e., edge).
    :rtype: ``int``

    :Example:

    Take the following lower border in an image:

    .. code-block:: python

             --- Image Here ---
        _____________________________
        The quick brown fox jumped
        over jumps over the lazy dog
        -----------------------------

    One pass of this procedure should produce:

    .. code-block:: python

             --- Image Here ---
        _____________________________
        The quick brown fox jumped
        -----------------------------

    Following this up with a second pass should yeild:

    .. code-block:: python

             --- Image Here ---
        _____________________________

    """
    first_pass_rslt = _lower_bar_detection(image_array, lower_bar_search_space, signal_strength_threshold)
    
    if lower_bar_second_pass:
        second_pass_rslt = _lower_bar_detection(image_array,
                                                lower_bar_search_space,
                                                signal_strength_threshold,
                                                cfloor=first_pass_rslt)
    else:
        second_pass_rslt = None

    if second_pass_rslt is None:
        return first_pass_rslt
    else:
        return second_pass_rslt


def border_detection(image,
                     signal_strength_threshold=0.25,
                     min_border_separation=0.15,
                     lower_bar_search_space=0.9,
                     report_signal_strength=False,
                     rescale_input_ndarray=True,
                     lower_bar_second_pass=True):
    """

    Detects the borders and lower bar in an image.

    At a high level, this algorithm works as follows:
       1. Along a given axis (rows or columns), vectors
          with standard deviation approximately equal to 0 are replaced with zero vectors. (a).
       2. Values are averaged along this same axis.
          This produces a signal (which can be visualized as a line graph). (b).
       3. The median value for this signal is computed. (c).
       4. The ``n`` points for which are the furthest, in absolute value, from the median are selected.
       5. The signal strength of the ``n`` points is quantified using percent error, where the median value
          is used as the expected value.
       6. Candidates for border pairs (e.g., left and right borders) are then weighed based on three lines of evidence.
          Namely, their signal strength, how separated they are and their absolute distance
          from the image's midline (about the corresponding axis). If a candidate fails to meet any of these criteria,
          it is rejected.
       7. The evidence for a lower bar concerns only its signal strength, though only an area of image below a given
          height is considered when trying to locate it. A double pass, the default, will try a second time to find
          another lower bar (for reasons explained in the docstring for the ``lower_bar_analysis()`` function).
          Regardless of whether or not the second pass could find a second edge, all of the edges detect are averaged
          and returned as an integer. If no plausible borders could be found, ``None`` is returned.

    (a) This reduces the muffling effect that areas with solid color can have on step 2.
    (b) Large inflections after areas with little change suggest a transition from a solid background to an image.
    (c) The median is used here, as opposed to the average, because it is more robust against outliers.

    :param image: a path to an image or an image represented as a 2D ndarray.

                 .. warning::

                        If a ``ndarray`` is passed, it should be the output of the
                        ``biovida.images.image_tools.load_image_rescale()`` function.
                        Without this preprocessing, this function's stability is not assured.

    :type image: ``str`` or ``2D ndarray``
    :param signal_strength_threshold: a value between 0 and 1 specify the signal strength required
                                      for an area required to be considered a 'lower bar'.
                                      Internally, this is measured as a location deviation from
                                      the median signal strength of the average image.
    :type signal_strength_threshold: ``float``
    :param min_border_separation: a value between 0 and 1 that determines the proportion of the axis
                                  that two edges must be separated for them to be considered borders.
                                  (i.e., ``axis_size`` * ``min_border_separation``)
    :type min_border_separation: ``float``
    :param lower_bar_search_space: a value between 0 and 1 specify the proportion of the image
                                   to search for a lower bar (e.g., 0.90). Set to ``None`` to disable.
    :type lower_bar_search_space: ``float``
    :param report_signal_strength: if ``True`` include the strength of the signal suggesting the existence of an edge.
                                   Defaults to ``False``.
    :type report_signal_strength: ``bool``
    :param rescale_input_ndarray: if True, rescale a ``2D ndarray`` passed to ``image``.
    :param lower_bar_second_pass:  if ``True`` perform a second pass on the lower bar. Defaults to ``True``.
    :type lower_bar_second_pass: ``bool``
    :type rescale_input_ndarray: ``bool``
    :return: a dictionary of the form:
             ``{'vborder': (left, right) or None, 'hborder': (upper, lower) or None, 'hbar': int or None}``

            - 'vborder' gives the locations of vertical borders.

            - 'hborder' gives the locations of horizontal borders.

            - 'hbar' gives the location for the top of the horizontal bar at the bottom of the image.

    :rtype: ``dict``
    """
    if 'numpy' in str(type(image)):
        image_copy = image.copy()
        if rescale_input_ndarray:
            image_array = load_image_rescale(image_copy, gray_only=True)
        else:
            image_array = image_copy
    elif isinstance(image, str):
        image_array = load_image_rescale(image)
    else:
        raise ValueError("`image_array` must be either a path to an image or the image as a 2D ndarray.")

    d = dict.fromkeys(['vborder', 'hborder', 'hbar'], None)

    v_edge_candidates = edge_detection(image_array, axis=0)
    d['vborder'] = _weigh_evidence(v_edge_candidates,
                                   image_array.shape[1],      # Note: we have (rows, columns) = (height, width).
                                   signal_strength_threshold,
                                   min_border_separation)

    h_border_candidates = edge_detection(image_array, axis=1)

    # Run Analysis. This excludes final element in `h_border_candidates` as including a third element
    # is simply meant to deflect the pull of the lower bar, if present.
    d['hborder'] = _weigh_evidence(h_border_candidates[:2],
                                   image_array.shape[0],
                                   signal_strength_threshold,
                                   min_border_separation)

    d['hbar'] = lower_bar_analysis(image_array=image_array,
                                   lower_bar_search_space=lower_bar_search_space,
                                   signal_strength_threshold=signal_strength_threshold,
                                   lower_bar_second_pass=lower_bar_second_pass)

    if report_signal_strength:  # ToDo: not working for 'hbar'.
        return d
    else:
        return {k: tuple([i[0] for i in v]) if isinstance(v, list) else (v[0] if isinstance(v, (list, tuple)) else v)
                for k, v in d.items()}


def _lines_plotter(path_to_image):
    """

    Visualizes the default ``border_detection()`` settings.

    :param path_to_image: path to the image
    :type path_to_image: ``str``
    """
    from matplotlib import pyplot as plt
    from matplotlib import collections as mc

    image = load_image_rescale(path_to_image)
    analysis = {k: v for k, v in border_detection(image=image).items() if v is not None}

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
