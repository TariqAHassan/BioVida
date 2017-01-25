"""

    Template Matching
    ~~~~~~~~~~~~~~~~~

"""
# Notes:
#     Powered by Fast Normalized Cross-Correlation.
#     See: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.match_template.
#     Here, this algorithm has been bootstrapped to make it robust against variance in scale.


# Imports
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from skimage.feature import match_template
from skimage.color.colorconv import rgb2gray


def _arange_one_first(start, end, step, precision=1):
    """

    Wrapper for np.arange where the number '1' is always first.
    Note: zero will be removed, if created, from the final array.

    :param start: the starting value.
    :type start: ``int``
    :param end: the ending value.
    :type end: ``int``
    :param step: the step size.
    :type step: ``int``
    :param precision: number of decimals to evenly round the array.
    :type precision:  ``int``
    :return: an array created by arange where the number `1` is invariably the first element in the array.
    :rtype: ``ndarray``
    """
    arr = np.around(np.arange(start, end, step), precision)
    arr = arr[arr != 0]
    return np.append(1, arr[arr != 1.0])


def _cropper(base, h_prop=0.5, v_prop=1/3):
    """

    Crops an image horizontally and vertically.

    :param base: an image represented as a 2D array.
    :type base: ``2D ndarray``
    :param h_prop: the proportion of the image to remove with respect to the x axis.
    :type h_prop: ``float``
    :param v_prop: the proportion of the image to remove with respect to the y axis.
    :type v_prop: ``float``
    :return: a cropped image as a 2D array
    :rtype: ``2D ndarray``
    """
    # Crop to the left
    hcrop = int(base.shape[0] * h_prop)
    hcrop_base = base[:, hcrop:]

    # Crop to the top
    hcrop_vcrop_base = hcrop_base[0:int(hcrop_base.shape[1] * v_prop)]

    return hcrop_vcrop_base


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


def _robust_match_template_loading(img, param_name):
    """

    Loads images for `robust_match_template()`

    :param img: a path to an image or the image as a 2D array
    :type img: ``str`` or ``2D ndarray``
    :param param_name: the name of the parameter which is being loaded (i.e., `pattern_img` or `base_img`.
    :type param_name: ``str``
    :return: an image as an array.
    :rtype: ``2D ndarray``
    """
    if 'ndarray' in str(type(img)):
        return img
    elif isinstance(img, str):
        return imread(img, flatten=True)
    else:
        raise ValueError("`{0}` must either be a `ndarray` or a path to an image.".format(param_name))


def _min_base_rescale(base, pattern, base_rescale, round_to=3):
    """

    Corrects ``base_rescale`` in instances where it would result
    in the ``base`` image being rescaled to a size which is smaller than the ``pattern`` image.

    Notes:

        - if ``abs(base_rescale[1] - base_rescale[0]) < step size`` at the end of the this function,
           only the unit transformation will take place in ``_matching_engine()``.

        - this function cannot handle the rare case where the pattern is larger than the base.

    :param base:
    :type base: ``2D ndarray``
    :param pattern:
    :type pattern: ``2D ndarray``
    :param base_rescale:
    :type base_rescale: ``tuple
    :param round_to: how many places after the decimal to round to. Defaults to 3.
    :type round_to: ``int``
    :return: ``base_rescale`` either 'as is' or updated to prevent the case outlined in this function's description.
    :rtype: ``tuple``
    """
    # Pick the limiting axis in the base image (smallest)
    smallest_base_axis = min(base.shape)

    # Pick the limiting axis in the base (largest)
    size_floor = max(pattern.shape)

    # Compute the smallest scalar
    min_scalar_for_base = float(np.around(size_floor / smallest_base_axis, round_to))

    # Convert `base_rescale` into a list
    base_rescale = list(base_rescale)

    # Move the rescale into valid range, if needed
    if base_rescale[0] < min_scalar_for_base:
        base_rescale[0] = min_scalar_for_base
    if base_rescale[1] < min_scalar_for_base:
        base_rescale[1] = base_rescale[1] + min_scalar_for_base

    return tuple(base_rescale)

def _matching_engine(base, pattern, base_rescale, h_crop, end_search_threshold):
    """

    Runs ``skimage.feature.match_template()`` against ``base`` for a given pattern
    at various resizings of the base image.

    :param base: the base image (typically cropped)
    :type base:
    :param pattern:
    :type pattern:
    :param base_rescale: ``(starting scalar, ending scalar, step size)``
    :type base_rescale: ``tuple``
    :param h_crop:
    :type h_crop:
    :param end_search_threshold: if a match of this quality is found, end the search. Set ``None`` to disable.
    :type end_search_threshold: ``float`` or  ``None``
    :return:
    :rtype: ``dict``
    """
    # Apply tool to ensure the base will always be larger than the pattern
    start, end, step = _min_base_rescale(base, pattern, base_rescale, round_to=3)

    match_dict = dict()
    for scale in _arange_one_first(start=start, end=end, step=step):
        # Rescale the image
        scaled_cropped_base = imresize(base, scale, interp='lanczos')

        # Run the template matching algorithm
        template_match_analysis = match_template(image=scaled_cropped_base, template=pattern)

        # Get the top left corner of the match and the match quality
        top_left, match_quality = _best_guess_location(template_match_analysis, scaling=scale)
        top_left_adj = top_left + np.array([h_crop, 0])

        # Work out the bottom right
        bottom_right = top_left_adj + np.floor(np.array(pattern.shape)[::-1] / scale)

        # Record
        match_dict[scale] = (list(top_left_adj), list(bottom_right), match_quality)

        # Test if the match was sufficently strong
        if isinstance(end_search_threshold, (int, float)):
            if match_quality >= end_search_threshold:
                break

    return match_dict


def _corners_calc(top_left, bottom_right):
    """

    Compute a dict. with a bounding box derived from
    a top left and top right corner

    :param top_left_: tuple of the form: (x, y).
    :param top_left_: ``tuple``
    :param bottom_right_: tuple of the form: (x, y)
    :param bottom_right_: ``tuple``
    :return: a dictionary with the following keys: 'top_left', 'top_right', 'bottom_left' and 'bottom_right'.
             Values are keys of the form (x, y).
    :rtype: ``dict``
    """
    d = {'top_left': top_left,
         'top_right': (bottom_right[0],top_left[1]),
         'bottom_left': (top_left[0], bottom_right[1]),
         'bottom_right': bottom_right}
    return {k: tuple(map(int, v)) for k, v in d.items()}


def robust_match_template(base_img, pattern_img, base_rescale=(0.5, 2.0, 0.1), end_search_threshold=0.875, base_cropping=(0.5, 1/3)):
    """

    Search for a pattern image in a base image using a algorithm which is robust
    against variation in the size of the pattern in the base image.

    Method: Fast Normalized Cross-Correlation.

    Limitations:

        - Cropping is limited to the the top left of the base image. The could be circumvented by setting
          ``base_cropping = (1, 1)`` and cropping ``base_img`` oneself.

    :param base_img:
    :type base_img:
    :param pattern_img:
    :type pattern_img:
    :param end_search_threshold: if a match of this quality is found, end the search. Set ``None`` to disable.
    :type end_search_threshold: ``float`` or  ``None``
    :param base_cropping: ``(horizontal (x), vertical (y))``
    :type base_cropping:
    :return:
    :rtype:
    """
    # Load the pattern images
    pattern = _robust_match_template_loading(pattern_img, "pattern_img")

    # Load the base
    base = _robust_match_template_loading(base_img, "base_img")

    # Crop the base
    h_crop = int(base.shape[0] * base_cropping[0])
    cropped_base = _cropper(base, h_prop=base_cropping[0], v_prop=base_cropping[1])

    # Search for matches
    match_dict = _matching_engine(cropped_base, pattern, base_rescale, h_crop, end_search_threshold)

    # Extract the best match
    best_match = max(list(match_dict.values()), key=lambda x: x[2])

    # Compute the bounding box
    bounding_box = _corners_calc(best_match[0], best_match[1])

    # Return the bounding box, match quality and the size of the base image
    return {"bounding_box": bounding_box, "match_quality":best_match[2], "base_img_size": base.shape}


def _box_show(base_img_path, pattern_img_path):
    """

    This function uses matplotlib to show the bounding box for the pattern.

    :param base_img_path: the path to the base image.
    :type base_img_path: ``str``
    :param pattern_img_path: the path to the pattern image.
    :type pattern_img_path: ``str``
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Load the Images
    base_img = imread(base_img_path, flatten=True)
    pattern_img = imread(pattern_img_path, flatten=True)

    # Run the analysis
    rslt = robust_match_template(base_img, pattern_img)

    # Extract the top left and top right
    top_left = rslt['bounding_box']['top_left']
    bottom_right = rslt['bounding_box']['bottom_right']

    # Compute the width and height
    width = abs(bottom_right[0] - top_left[0])
    height = abs(bottom_right[1] - top_left[1])

    # Show the base image
    fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 8), sharex=True, sharey=True)
    ax1.imshow(base_img, 'gray')

    # Add the bounding box
    ax1.add_patch(patches.Rectangle(top_left, width, height, fill=False, edgecolor="red"))
    fig.show()
































































