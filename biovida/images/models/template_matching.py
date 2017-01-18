"""

    Template Matching
    ~~~~~~~~~~~~~~~~~

"""
# Notes:
#     Powered by Fast Normalized Cross-Correlation.
#     See: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.match_template.
#
#     Here, this algorithm has been bootstrapped to make it robust against variance in scale.
#
#     ToDo: This code needs to be refactored.


# Imports
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from skimage.feature import match_template
from skimage.color.colorconv import rgb2gray


def _bounds_t_match(pattern_shape, base_shape, prop_scale, scaling_lower_limit):
    """

    Defines a vector over which to iterate when scaling the pattern shape in _scale_invar_match_template.
    This algorithm *should* solve the general form of the problem.

    Motivation:
      a. pattern is smaller than base
          - want to known x, for P * x before P large <- TL (too large)
          - set lower bound to be percentage of TL
          - compute sequence from [lower bound, upper bound]
      b. pattern is larger than base
          - want to known x for P / x until P is suffcently small <- SS (sufficently small)
          - set upper bound to be percentage of SS.
          - compute sequence from [lower bound, upper bound]

    :param pattern_shape: the pattern to search for. Form: (nrows, ncols).
    :param base_shape: the image to search for the patten in. Form: (nrows, ncols).
    :param prop_scale: defines the number of steps to take between the bounds. Must be a float on (0, 1).
                       Note: this interval is not inclusive.
    :param scaling_lower_limit: smallest acceptable bound. Set to 0.0 to disable.
    :return: an array of values to scale the pattern by in `_scale_invar_match_template()`.
    :rtype: ``ndarray``
    """
    # WARNING: IMG.shape is given as: (ROWS, COLUMNS) = (HEIGHT, WIDTH).
    def bounds_clean(arr):
        if scaling_lower_limit >= 1 and 1 not in arr:
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
        lower_bound = prop_scale
        step = (upper_bound - lower_bound) * prop_scale
        bounds_array = np.arange(lower_bound, upper_bound, step)

    # Case 3 (pattern > base): The pattern must only shrink
    # More formally: nrows(pattern) > nrows(base) OR ncols(pattern) > ncols(base)
    elif any(pattern_shape[i] > base_shape[i] for i in range(2)):
        upper_bound = 1 / np.max(np.array(pattern_shape) / np.array(base_shape))
        lower_bound = upper_bound * prop_scale
        bounds_array = np.linspace(upper_bound, lower_bound, prop_scale * 100)

    # Impose lower limit and add identity transformation.
    return np.append(bounds_clean(bounds_array), 1)


def _best_guess_location(match_template_result, upscale=1):
    """

    Takes the result of skimage.feature.match_template() and returns (top left x, top left y)

    :param match_template_result:
    :return:
    """
    ij = np.unravel_index(np.argmax(match_template_result), match_template_result.shape)
    x, y = ij[::-1]
    return np.ceil(np.array((x, y)) / float(upscale)), match_template_result.max()


def _corners_calc(top_left_, bottom_right_):
    """

    Compute a dict. with

    :param top_left_:
    :param bottom_right_:
    :return:
    """
    d = {
        'top_left': top_left_,
        'top_right': (bottom_right_[0], top_left_[1]),
        'bottom_left': (top_left_[0], bottom_right_[1]),
        'bottom_right': bottom_right_}
    return {k: tuple(map(int, v)) for k, v in d.items()}

def _scale_invar_match_template_output(d):
    """

    :param d:
    :return:
    """
    if not len(list(d.keys())):
        return None

    best_match = max(d, key=lambda x: d.get(x)[0])
    return {"match_quality": d[best_match][0],
            "box": _corners_calc(top_left_=d[best_match][1][0], bottom_right_=d[best_match][1][1])}


def _scale_invar_match_template(pattern_img
                                , base_img
                                , base_top_cropping
                                , prop_scale
                                , scaling_lower_limit
                                , base_resize
                                , end_search_threshold):
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
    :param prop_scale:
    :param scaling_lower_limit:
    :return: see robust_match_template()
    :rtype: ``dict``
    """
    # ToDo: update algorithm summary as it is no longer accurate.
    # Top x (1/3) for the base image and crop accordingly
    top_of_base_img = int(base_img.shape[0] * base_top_cropping)
    base_img_cropped = base_img[:top_of_base_img]

    # Compute the cropped size of the base_img
    base_img_scaled = (base_img.shape[0], base_img.shape[1] / top_of_base_img)

    # Compute an array of values to scale the pattern by
    bounds_vect = _bounds_t_match(pattern_img.shape[0:2], base_img_scaled, prop_scale, scaling_lower_limit)

    d = dict()
    for pattern_scale in bounds_vect:
        pattern_img_scaled = rgb2gray(imresize(pattern_img, pattern_scale, interp='lanczos'))
        if all(pattern_img_scaled.shape[i] < base_img_cropped.shape[i] for i in range(2)):  # ToDo: Shouldn't be needed.
            # Run the match_template algorithm
            result = match_template(image=base_img_cropped, template=pattern_img_scaled)

            # Top Left Corner of the bounding box
            top_left, match_quality = _best_guess_location(result, upscale=base_resize)

            # Note the need for pattern_img_scaled.shape to go from (HEIGHT, WIDTH) to (WIDTH, HEIGHT).
            bottom_right = top_left + np.array(pattern_img_scaled.shape)[::-1]

            d[pattern_scale] = (match_quality, (tuple(top_left), tuple(bottom_right)))

            if match_quality >= end_search_threshold:
                break

    return _scale_invar_match_template_output(d)


def _arrange_one_first(bounds_and_steps):
    """

    Function to run np.arrange and put the number '1' first.

    :param bounds_and_steps:
    :return:
    """
    start, end, step = bounds_and_steps if bounds_and_steps is not None else (1, 1, 1)
    l = np.arange(start, end, step)
    return np.append(1, l[l != 1])


def _robust_match_template_loading(img, param_name):
    """

    :param img:
    :param param_name:
    :return:
    """
    if 'ndarray' in str(type(img)):
        return img
    elif isinstance(img, str):
        return imread(img, flatten=True)
    else:
        raise ValueError("`{0}` must either be a `ndarray` or a path to an image.".format(param_name))


def robust_match_template(pattern_img
                          , base_img
                          , base_top_cropping=0.14
                          , prop_scale=0.075
                          , scaling_lower_limit=0.25
                          , end_search_threshold=0.875
                          , base_resizes=(1.25, 2.75, 0.25)):
    """

    Search for a pattern image in a base image using a algorithm which is robust
    against variation in the size of the pattern in the base image.

    Method: Fast Normalized Cross-Correlation.

    :param pattern_img:

                    ..warning:

                        If a `ndarray` is passed, it must be preprocessed with
                        ``scipy.misc.imread(pattern_img, flatten=True)``

    :param pattern_img: ``str`` or ``ndarray``
    :param base_img:

            ..warning:

                If a `ndarray` is passed, it must be preprocessed with
                ``scipy.misc.imread(base_img, flatten=True)``

    :param base_img: ``str`` or ``ndarray``
    :param base_top_cropping:
    :param prop_scale:
    :param scaling_lower_limit:
    :param end_search_threshold: if a match of this quality is found, end the search.
                                  Set to any number greater than 1 to disable.
    :param base_resizes: scaling of the base image. Tuple of the form (start scalar, end scalar, step size).
    :param base_img: if `None` the `
    :type base_img: ``None``, ``ndarray`` or ``str``
    :return: tuple of the form (best_match_dict, base image shape (x, y)). the 'best_match_dict' is of the form:
                     {match_quality: value between 0 and 1 (inclusive),
                     'box': {'bottom_right': (x, y), 'top_right': (x, y), 'top_left': (x, y), 'bottom_left': (x, y)}}
    :rtype: ``tuple``
    """
    # Load the Images
    pattern_img_raw = _robust_match_template_loading(pattern_img, "pattern_img")
    base_img_raw = _robust_match_template_loading(base_img, "base_img")

    attemps = list()
    for base_resize in _arrange_one_first(base_resizes):
        base_img = imresize(base_img_raw, base_resize)
        current_match = _scale_invar_match_template(pattern_img_raw,
                                                    base_img,
                                                    base_top_cropping,
                                                    prop_scale,
                                                    scaling_lower_limit,
                                                    base_resize,
                                                    end_search_threshold)

        if isinstance(current_match, dict):
            attemps.append(current_match)
            if current_match['match_quality'] >= end_search_threshold:
                break

    single_best_match = max(attemps, key=lambda x: x['match_quality'])
    return single_best_match, base_img.shape[::-1]







































