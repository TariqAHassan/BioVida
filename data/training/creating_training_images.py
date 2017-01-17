"""

    Tools to Create Training Data for Image Recognition
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import random
import warnings
import numpy as np
from random import randint
from tqdm import tqdm
from PIL import Image

from data.training.temp import (arrow_path,
                                valid_mri_path,
                                grid_save_location,
                                arrow_save_location)

# warnings.filterwarnings("error")

# ------------------------------------------------------------------------------------------
# General Tools
# ------------------------------------------------------------------------------------------


def resize_image(img, scalar=1/3):
    """

    :param image:
    :param scalar:
    :return:
    """
    width, height = img.size
    img = img.resize((int(width * scalar), int(height * scalar)))
    return img


def random_tuple_in_range(tpl, border_buffer=0.385):
    """

    :param tpl:
    :param border:
    :return:
    """
    w, h = tpl
    rand_x = randint(int(w * border_buffer), int(w * (1 - border_buffer)))
    rand_y = randint(int(h * border_buffer), int(h * (1 - border_buffer)))
    return rand_x, rand_y


def random_crop(img, choice_override=False):
    """

    :param img:
    :param choice_override:
    :return:
    """
    w, h = img.size
    w2, h2 = np.round(np.array(img.size) / 2).astype(int)

    if choice_override:
        choice = choice_override
    else:
        choice = randint(0, 2)
    side = randint(0, 1)

    if choice == 0:
        return img
    elif choice == 1: # crop w.r.t. width
        if side == 0:
            return img.crop((0, 0, w2, h))
        elif side == 1:
            return img.crop((w2, 0, w, h))
    elif choice == 2: # crop w.r.t. height
        if side == 0:
            return img.crop((0, 0, w, h2))
        elif side == 1:
            return img.crop((0, h2, w, h))


def random_stretch(img, stretch_by):
    """

    :param img:
    :param stretch_by:
    :return:
    """
    w, h = img.size
    scale_by = random.uniform(stretch_by[0], stretch_by[1])
    choice = randint(0, 2)

    if choice == 0:
        return img
    elif choice == 1:
        return img.resize((int(w * scale_by), h))
    elif choice == 2:
        return img.resize((w, int(h * scale_by)))


def open_muliple_and_random_crop(image_list):
    """

    :param image_list:
    :return:
    """
    to_crop_or_not_to_crop = randint(0, 1)
    if to_crop_or_not_to_crop == 0:
        return [Image.open(i) for i in image_list]
    else:
        return [random_crop(Image.open(i), randint(1, 2)) for i in image_list]


# ------------------------------------------------------------------------------------------
# Arrows
# ------------------------------------------------------------------------------------------


# Random valid MRI as background
# Random arrow
#     - location in image
#     - random size (in some range)
#     - random rotation

# Load MRIs
valid_mris = [os.path.join(valid_mri_path, i) for i in os.listdir(valid_mri_path) if i.endswith(".png")]

# Load arrows
arrows_raw = [i for i in os.listdir(arrow_path) if i.endswith(".png")]
sorted_arrows = sorted(arrows_raw, key=lambda x: int(x.split("_")[0]))
arrows = [os.path.join(arrow_path, i) for i in sorted_arrows]


def arrow_back_foreground_mash(background_path
                               , foreground_path
                               , foreground_scale_range=(0.1, 0.275)
                               , foreground_stretch_by=(0.8, 2.0)
                               , location_border_buffer=0.2):
    """

    :param background_path:
    :param foreground_path:
    :param foreground_scalar:
    :param location_border_buffer:
    :return:
    """
    # Load a Background
    background = Image.open(background_path).convert("RGBA")

    # Randomly rescale background
    background = resize_image(background, random.uniform(0.8, 1.2))

    # Randomly crop background
    background = random_crop(background)

    # Load the foreground and convert to grayscale
    foreground = Image.open(foreground_path).convert("RGBA")

    for _ in range(1, randint(2, 4)):
        # Randomly stretch about a given axis
        foreground = random_stretch(foreground, foreground_stretch_by)

        # Change contrast
        foreground = foreground.point(lambda p: p * random.uniform(0.87, 0.92))

        # Scale the foreground by a random amount in some range
        foreground_scalar = random.uniform(foreground_scale_range[0], foreground_scale_range[1])

        # Compute the correct scalar to make foreground `foreground_scalar` times
        # the size of the smallest axis in the background.
        scale_image_by = (min(background.size) * (foreground_scalar)) / max(foreground.size)

        # Rescale the foreground and rotate
        foreground = resize_image(foreground, scale_image_by).rotate(randint(0, 360))

        # Random location to place the image
        random_background_loc = random_tuple_in_range(background.size, location_border_buffer)

        # Paste the foreground onto the background
        background.paste(foreground, random_background_loc, foreground)

    return background


def arrow_masher(background_options, foreground_options, name, save_location):
    """

    :param background_options:
    :param foreground_options:
    :return:
    """
    # Select random element
    bg = random.choice(background_options)
    fg = random.choice(foreground_options)

    # Compute mash and convert to grayscale
    rslt = arrow_back_foreground_mash(bg, fg).convert("LA")

    # Save to disk
    rslt.save("{0}/{1}.png".format(save_location, name))


def arrow_creator(background_options, foreground_options, n, general_name, save_location):
    """

    :param background_options:
    :param foreground_options:
    :param n:
    :param save_location:
    :return:
    """
    for i in tqdm(range(1, n+1)):
        arrow_masher(background_options, foreground_options, name="{0}_{1}".format(i, general_name), save_location=save_location)


# Create the arrow training data
arrow_creator(valid_mris, arrows, n=10000, save_location=arrow_save_location)

# ------------------------------------------------------------------------------------------
# Grids
# ------------------------------------------------------------------------------------------

# Probabilities for
# Image Creation:

# ----------------------------------------------------------------
# ImgShape              ID                 Likelihood
# ----------------------------------------------------------------
# H stacking            0                  0.25
# V stacking            1                  0.25
# HB stacking           2                  0.30 (above/below)
# VB stacking           3                  0.20 (side-by-side)
# ----------------------------------------------------------------
# ImgShape          N images              Likelihood
# ----------------------------------------------------------------
# H stacking        2, 3, 4               (1/3, 1/3, 1/3)
# V stacking        2, 3                  (1/2, 1/2)
# HB stacking       2, 4, 6, 8            (1/4, 1/4, 1/4, 1/4)
# VB stacking       2, 4, 6, 8            (1/4, 1/4, 1/4, 1/4)
# ----------------------------------------------------------------


def img_stacker(image_list, stacker=np.hstack, min_shape_override=None):
    """

    Combine Images horizontally.

    :param image_list:
    :return:
    """
    # see: http://stackoverflow.com/a/30228789/4898004.
    if min_shape_override is not None:
        imgs = image_list
        min_shape = min_shape_override
    else:
        imgs = open_muliple_and_random_crop(image_list)
        # imgs = [Image.open(i) for i in image_list]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]

    # Combine
    return stacker((np.asarray(i.resize(min_shape)) for i in imgs))


def side_by_side_stacker(image_list, stacker_a, stacker_b):
    """

    :param image_list:
    :return:
    """
    # Open the images
    # imgs = [Image.open(i) for i in image_list]
    imgs = open_muliple_and_random_crop(image_list)

    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]

    # Generate the first grid
    grid_1 = img_stacker(image_list=imgs[:int(len(imgs)/2)], stacker=stacker_a, min_shape_override=min_shape)

    # Generate the second grid
    grid_2 = img_stacker(image_list=imgs[int(len(imgs)/2):], stacker=stacker_a, min_shape_override=min_shape)

    # Stack side-by-side
    return stacker_b((grid_1, grid_2))


def grid_masher(all_img_options, name, save_location):
    """

    :return:
    """
    # Determine the general shape of the grid
    img_shape = np.random.choice(4, 1, p=[0.25, 0.25, 0.30, 0.20])[0]

    # Determine possibilities for the number of images based on img_shape.
    if img_shape == 0:
        choices, weights = [2, 3, 4], [1/3, 1/3, 1/3]
    elif img_shape == 1:
        choices, weights = [2, 3], [1/2, 1/2]
    elif img_shape == 2:
        choices, weights = [2, 4, 6, 8], [1/4, 1/4, 1/4, 1/4]
    elif img_shape == 3:
        choices, weights = [4, 6, 8], [1/3, 1/3, 1/3]

    # Pick random number of images, given `img_shape`.
    n_images = np.random.choice(choices, 1, weights)[0]

    # Extract some the randomly specified number of images
    grid_images = np.random.choice(all_img_options, n_images, replace=False)

    # Stack
    if img_shape == 0:
        grid = img_stacker(grid_images)
    elif img_shape == 1:
        grid = img_stacker(grid_images, stacker=np.vstack)
    elif img_shape == 2:
        grid = side_by_side_stacker(grid_images, stacker_a=np.hstack, stacker_b=np.vstack)
    elif img_shape == 3:
        grid = side_by_side_stacker(grid_images, stacker_a=np.vstack, stacker_b=np.hstack)

    # Image.fromarray(grid).show()
    rslt = Image.fromarray(grid)

    # Convert to an image and Save to disk
    rslt.save("{0}/{1}.png".format(save_location, name))


def grid_creator(all_img_options, n, general_name, save_location):
    """

    :param all_img_options:
    :param n:
    :param save_location:
    :return:
    """
    for i in tqdm(range(1, n+1)):
        grid_masher(valid_mris, name="{0}_{1}".format(i, general_name), save_location=save_location)


# Create the grid training data
grid_creator(valid_mris, n=10000, save_location=grid_save_location)









































































