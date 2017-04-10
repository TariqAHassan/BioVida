# coding: utf-8

"""

    Template Matching
    ~~~~~~~~~~~~~~~~~

"""
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from skimage.feature import match_template


# Notes:
#     See: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.match_template.
#     Here, this algorithm has been bootstrapped to make it robust against variance in scale.

# ToDo: the solution implemented here, while it works somewhat well, is lacking.
#       The correct approach is to systematically crop the image as long as the signal
#       continues to increases, up to some ceiling value.


def _arange_one_first(start, end, step, precision=1):
    """

    Wrapper for ``numpy.arange()`` where the number '1' is always first.
    Note: zero will be removed, if created, from the final array.

    :param start: the starting value.
    :type start: ``int``
    :param end: the ending value.
    :type end: ``int``
    :param step: the step size.
    :type step: ``int``
    :param precision: number of decimals to evenly round the array.
    :type precision:  ``int``
    :return: an array created by ``numpy.arange()`` where the number `1` is invariably the first element in the array.
    :rtype: ``ndarray``
    """
    arr = np.around(np.arange(start, end, step), precision)
    arr = arr[arr != 0]
    return np.append(1, arr[arr != 1.0])


def _cropper(base, v_prop, h_prop):
    """

    Crops an image horizontally and vertically.

    Notes:

        - increasing ``h_prop`` increases the amount of the image's left side removed.

        - increasing ``v_prop`` increases the amount of the lower part of the image removed.

    :param base: an image represented as a 2D array.
    :type base: ``2D ndarray``
    :param v_prop: the proportion of the image to remove with respect to its height.
    :type v_prop: ``float``
    :param h_prop: the proportion of the image to remove with respect to its width.
    :type h_prop: ``float``
    :return: a cropped image as a 2D array
    :rtype: ``2D ndarray``
    """
    # Note: image.shape = (height, width).

    # Crop to the left
    hcrop_base = base[:, int(base.shape[1] * h_prop):]

    # Crop with respect to height
    return hcrop_base[:int(base.shape[0] * v_prop)]


def _best_guess_location(match_template_result, scaling=1):
    """

    Takes the result of skimage.feature.match_template() and returns (top left x, top left y)
    by selecting the item in ``match_template_result`` with the strongest signal.

    :param match_template_result: the output of from skimage.feature import match_template.
    :type match_template_result: ``ndarray``
    :return: the upper left for location of the strongest peak and the match quality. Form: ``((x, y), match quality)``.
    :rtype: ``tuple``
    """
    x, y = np.unravel_index(np.argmax(match_template_result), match_template_result.shape)[::-1]
    return np.ceil(np.array((x, y)) / float(scaling)), match_template_result.max()


def _robust_match_template_loading(image, param_name):
    """

    Loads images for ``robust_match_template()``.

    :param image: a path to an image or the image as a 2D array
    :type image: ``str`` or ``2D ndarray``
    :param param_name: the name of the parameter which is being loaded (i.e., `pattern_image` or `base_image`.
    :type param_name: ``str``
    :return: an image as an array.
    :rtype: ``2D ndarray``
    """
    if 'ndarray' in str(type(image)):
        return image
    elif isinstance(image, str):
        return imread(image, flatten=True)
    else:
        raise ValueError("`{0}` must either be a `ndarray` or a path to an image.".format(param_name))


def _min_base_rescale(base, pattern, base_resizes, round_to=3):
    """

    Corrects ``base_resizes`` in instances where it would result
    in the ``base`` image being rescaled to a size which is smaller than the ``pattern`` image.

    Notes:

        - if ``abs(base_resizes[1] - base_resizes[0]) < step size`` at the end of the this function,
           only the unit transformation will take place in ``_matching_engine()``.

        - this function cannot handle the rare case where the pattern is larger than the base.

    :param base: the base image as a 2D array.
    :type base: ``2D ndarray``
    :param pattern: the pattern image as a 2D array.
    :type pattern: ``2D ndarray``
    :param base_resizes: the range over which to rescale the base image.Define as a tuple of the
                         form ``(start, end, step size)``.
    :type base_resizes: ``tuple``
    :param round_to: how many places after the decimal to round to. Defaults to 3.
    :type round_to: ``int``
    :return: ``base_resizes`` either 'as is' or updated to prevent the case outlined in this function's description.
    :rtype: ``tuple``
    """
    # ToDo: Does not always block base < pattern.

    # Pick the limiting axis in the base image (smallest)
    smallest_base_axis = min(base.shape)

    # Pick the limiting axis in the base (largest)
    size_floor = max(pattern.shape)

    min_scalar_for_base = float(np.around(size_floor / smallest_base_axis, round_to))
    base_resizes = list(base_resizes)

    # Move the rescale into valid range, if needed
    if base_resizes[0] < min_scalar_for_base:
        base_resizes[0] = min_scalar_for_base
    if base_resizes[1] < min_scalar_for_base:
        base_resizes[1] += min_scalar_for_base

    return tuple(base_resizes)


def _matching_engine(base, pattern, base_resizes, base_image_cropping, end_search_threshold):
    """

    Runs ``skimage.feature.match_template()`` against ``base`` for a given ``pattern``
    at various sizes of the base image.

    :param base: the base image (typically cropped)
    :type base: ``2D ndarray``
    :param pattern: the pattern image
    :type pattern: `2D ndarray``
    :param base_resizes: the range over which to rescale the base image.Define as a tuple of the
                         form ``(start, end, step size)``.
    :type base_resizes: ``tuple``
    :param base_image_cropping: see ``robust_match_template()``.
    :type base_image_cropping: ``tuple``
    :param end_search_threshold: if a match of this quality is found, end the search. Set ``None`` to disable.
    :type end_search_threshold: ``float`` or  ``None``
    :return: a dictionary of matches made by the ``skimage.feature.match_template()`` function
             with the base image scaled by different amounts (represented by the keys).
             The values are ``tuples`` of the form ``(top left corner, bottom right corner, match quality)``.
    :rtype: ``dict``
    """
    # Crop the base image
    base_h_crop = int(base.shape[1] * base_image_cropping[1])
    cropped_base = _cropper(base, v_prop=base_image_cropping[0], h_prop=base_image_cropping[1])

    # Apply tool to ensure the base will always be larger than the pattern
    start, end, step = _min_base_rescale(cropped_base, pattern, base_resizes, round_to=3)

    match_dict = dict()
    for scale in _arange_one_first(start=start, end=end, step=step):
        # Rescale the image
        scaled_cropped_base = imresize(cropped_base, scale, interp='lanczos')

        # ToDo: this try/except should not be needed.
        try:
            template_match_analysis = match_template(image=scaled_cropped_base, template=pattern)

            top_left, match_quality = _best_guess_location(template_match_analysis, scaling=scale)
            top_left_adj = top_left + np.array([base_h_crop, 0])

            bottom_right = top_left_adj + np.floor(np.array(pattern.shape)[::-1] / scale)
            match_dict[scale] = (list(top_left_adj), list(bottom_right), match_quality)

            if isinstance(end_search_threshold, (int, float)):
                if match_quality >= end_search_threshold:
                    break
        except:
            pass

    return match_dict


def _corners_calc(top_left, bottom_right):
    """

    Compute a dict. with a bounding box derived from
    a top left and top right corner

    :param top_left: tuple of the form: (x, y).
    :param top_left: ``tuple``
    :param bottom_right: tuple of the form: (x, y)
    :param bottom_right: ``tuple``
    :return: a dictionary with the following keys: 'top_left', 'top_right', 'bottom_left' and 'bottom_right'.
             Values are keys of the form (x, y).
    :rtype: ``dict``
    """
    d = {'top_left': top_left,
         'top_right': (bottom_right[0], top_left[1]),
         'bottom_left': (top_left[0], bottom_right[1]),
         'bottom_right': bottom_right}
    return {k: tuple(map(int, v)) for k, v in d.items()}


def robust_match_template(pattern_image,
                          base_image,
                          base_resizes=(0.5, 2.5, 0.1),
                          end_search_threshold=0.875,
                          base_image_cropping=(0.15, 0.5)):
    """

    Search for a pattern image in a base image using a algorithm which is robust
    against variation in the size of the pattern in the base image.

    Method: Fast Normalized Cross-Correlation.

    Limitations:

    - Cropping is limited to the the top left of the base image. The can be circumvented by setting
      ``base_image_cropping=(1, 1)`` and cropping ``base_image`` oneself.

    - This function may become unstable in situations where the pattern image is larger than the base image.

    :param pattern_image: the pattern image.

            .. warning::

                    If a `ndarray` is passed to `pattern_image`, it *must* be preprocessed to be a 2D array,
                    e.g., ``scipy.misc.imread(pattern_image, flatten=True)``

    :type pattern_image: ``str`` or ``ndarray``
    :param base_image: the base image in which to look for the ``pattern_image``.

             .. warning::

                    If a `ndarray` is passed to `base_image`, it *must* be preprocessed to be a 2D array,
                    e.g., ``scipy.misc.imread(base_image, flatten=True)``

    :type base_image: ``str`` or ``ndarray``
    :param base_resizes: the range over which to rescale the base image.Define as a tuple of the
                         form ``(start, end, step size)``. Defaults to ``(0.5, 2.0, 0.1)``.
    :type base_resizes: ``tuple``
    :param end_search_threshold: if a match of this quality is found, end the search. Set equal to ``None`` to disable.
                                 Defaults to 0.875.
    :type end_search_threshold: ``float`` or  ``None``
    :param base_image_cropping: the amount of the image to crop with respect to the x and y axis.
                              form: ``(height, width)``. Defaults to ``(0.15, 0.5)``.
                          
    Notes:

    - Decreasing ``height`` will increase the amount of the lower part of the image removed.

    - Increasing ``width`` will increase the amount of the image's left half removed.

    - Cropping more of the base image reduces the probability that the algorithm getting confused.
      However, if the image is cropped too much, the target pattern itself could be removed.

    :type base_image_cropping: ``tuple``
    :return: A dictionary of the form: ``{"bounding_box": ..., "match_quality": ..., "base_image_shape": ...}``.

            - bounding_box (``dict``): ``{'bottom_right': (x, y), 'top_right': (x, y), 'top_left': (x, y), 'bottom_left': (x, y)}``.

            - match_quality (``float``): quality of the match.

            - base_image_shape (``tuple``): the size of the base image provided. Form: ``(width (x), height (y))``.

    :rtype: ``dict``
    """
    pattern = _robust_match_template_loading(pattern_image, "pattern_image")
    base = _robust_match_template_loading(base_image, "base_image")

    match_dict = _matching_engine(base, pattern, base_resizes, base_image_cropping, end_search_threshold)

    if len(list(match_dict.keys())):
        best_match = max(list(match_dict.values()), key=lambda x: x[2])
        bounding_box = _corners_calc(best_match[0], best_match[1])
        match_quality = best_match[2]
    else:
        bounding_box = None
        match_quality = None

    # Return the bounding box, match quality and the size of the base image
    return {"bounding_box": bounding_box, "match_quality": match_quality, "base_image_shape": base.shape[::-1]}


def _box_show(base_image_path, pattern_image_path):
    """

    This function uses matplotlib to show the bounding box for the pattern.

    :param base_image_path: the path to the base image.
    :type base_image_path: ``str``
    :param pattern_image_path: the path to the pattern image.
    :type pattern_image_path: ``str``
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Load the Images
    base_image = imread(base_image_path, flatten=True)
    pattern_image = imread(pattern_image_path, flatten=True)

    # Run the analysis
    rslt = robust_match_template(pattern_image, base_image)

    # Extract the top left and top right
    top_left = rslt['bounding_box']['top_left']
    bottom_right = rslt['bounding_box']['bottom_right']

    # Compute the width and height
    width = abs(bottom_right[0] - top_left[0])
    height = abs(bottom_right[1] - top_left[1])

    # Show the base image
    fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5), sharex=True, sharey=True)
    ax1.imshow(base_image, 'gray')

    # Add the bounding box
    ax1.add_patch(patches.Rectangle(top_left, width, height, fill=False, edgecolor="red"))
    fig.show()
