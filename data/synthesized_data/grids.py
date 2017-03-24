"""

    Grids
    ~~~~~

"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.synthesized_data.support_tools import (quality, open_multiple_and_random_crop)


# Probabilities for
# Image Creation:
# ----------------------------------------------------------------
# imageShape              ID                 Likelihood
# ----------------------------------------------------------------
# H stacking            0                  0.25
# V stacking            1                  0.25
# HB stacking           2                  0.30 (above/below)
# VB stacking           3                  0.20 (side-by-side)
# ----------------------------------------------------------------
# imageShape          N images              Likelihood
# ----------------------------------------------------------------
# H stacking        2, 3, 4               (1/3, 1/3, 1/3)
# V stacking        2, 3                  (1/2, 1/2)
# HB stacking       2, 4, 6, 8            (1/4, 1/4, 1/4, 1/4)
# VB stacking       2, 4, 6, 8            (1/4, 1/4, 1/4, 1/4)
# ----------------------------------------------------------------


def image_stacker(image_list, stacker=np.hstack, min_shape_override=None):
    """

    Combine Images horizontally.

    """
    # see: http://stackoverflow.com/a/30228789/4898004.
    if min_shape_override is not None:
        images = image_list
        min_shape = min_shape_override
    else:
        images = open_multiple_and_random_crop(image_list)
        # images = [Image.open(i) for i in image_list]
        min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]

    # Combine
    return stacker((np.asarray(i.resize(min_shape,  Image.ANTIALIAS)) for i in images))


def side_by_side_stacker(image_list, stacker_a, stacker_b):
    """

    :param image_list:
    :return:
    """
    # Open the images
    images = open_multiple_and_random_crop(image_list)

    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]

    # Generate the first grid
    grid_1 = image_stacker(image_list=images[:int(len(images)/2)], stacker=stacker_a, min_shape_override=min_shape)

    # Generate the second grid
    grid_2 = image_stacker(image_list=images[int(len(images)/2):], stacker=stacker_a, min_shape_override=min_shape)

    # Stack side-by-side
    return stacker_b((grid_1, grid_2))


def grid_masher(all_image_options):
    """

    :param all_image_options:
    :return:
    """
    # Determine the general shape of the grid
    image_shape = np.random.choice(4, 1, p=[0.25, 0.25, 0.30, 0.20])[0]

    # Determine possibilities for the number of images based on image_shape.
    if image_shape == 0:
        choices, weights = [2, 3, 4], [1/3, 1/3, 1/3]
    elif image_shape == 1:
        choices, weights = [2, 3], [1/2, 1/2]
    elif image_shape == 2:
        choices, weights = [2, 4, 6, 8], [1/4, 1/4, 1/4, 1/4]
    elif image_shape == 3:
        choices, weights = [4, 6, 8], [1/3, 1/3, 1/3]

    # Pick random number of images, given `image_shape`.
    n_images = np.random.choice(choices, 1, weights)[0]

    # Extract some the randomly specified number of images
    grid_images = np.random.choice(all_image_options, n_images, replace=False)

    # Stack
    if image_shape == 0:
        grid = image_stacker(grid_images)
    elif image_shape == 1:
        grid = image_stacker(grid_images, stacker=np.vstack)
    elif image_shape == 2:
        grid = side_by_side_stacker(grid_images, stacker_a=np.hstack, stacker_b=np.vstack)
    elif image_shape == 3:
        grid = side_by_side_stacker(grid_images, stacker_a=np.vstack, stacker_b=np.hstack)

    # Convert to a PIL image
    return Image.fromarray(grid)


def grid_masher_min_size(all_image_options, name, save_location, min_size=150, limit=250):
    """

    :param all_image_options:
    :param name:
    :param save_location:
    :param min_size:
    :param limit:
    :return:
    """
    image = grid_masher(all_image_options)

    c = 0
    while c <= limit and min(image.size) < min_size:
        image = grid_masher(all_image_options)
        c += 1

    image.save(os.path.join(save_location, "{0}.png".format(name)), quality=quality)


def grid_creator(all_image_options, start, end, general_name, save_location):
    """

    :param all_image_options:
    :param start:
    :param end:
    :param general_name:
    :param save_location:
    :return:
    """
    for i in tqdm(range(start+1, end+1)):
        grid_masher_min_size(all_image_options, name="{0}_{1}".format(i, general_name),
                             save_location=save_location)
