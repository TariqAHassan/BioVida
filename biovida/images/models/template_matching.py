"""

    Template Matching
    ~~~~~~~~~~~~~~~~~

"""
# Notes:
#     Powered by Fast Normalized Cross-Correlation.
#     See: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.match_template.
#
#     Here, his algorithm has been bootstrapped to make it robust against variance in scale.

# Imports
import numpy as np
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
          - want to known x, for P * x before P large <- TL (too large)
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
            'top_right': (bottom_right[0], top_left[1]),
            'bottom_left': (top_left[0], bottom_right[1]),
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















































