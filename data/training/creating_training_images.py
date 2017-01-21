"""

    Tools to Create Training Data for Image Recognition
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from random import randint

from data.training.my_file_paths import (cache_path,
                                         arrow_path,
                                         base_img_path,
                                         grid_save_location,
                                         arrow_save_location,
                                         ellipses_save_location)


# ------------------------------------------------------------------------------------------
# Define Images to use
# ------------------------------------------------------------------------------------------


# Define Images to Use
base_images = [os.path.join(base_img_path, i) for i in os.listdir(base_img_path) if i.endswith(".png")]


def base_image_record(images_used, cache_path, save_path):
    """
    
    Tool to create a csv of images saved.
    
    :param images_used: list of images
    :type images_used: ``list``
    :param cache_path: where the cache record is located.
    :type cache_path: ``str``
    :param save_path: where the record should be saved.
    :type save_path: ``str``
    """
    # Convert the list into a pandas dataframe
    images_used = pd.DataFrame(images_used).rename(columns={0: "ImageName"})
    images_used['ImageName'] = images_used['ImageName'].map(lambda x: x.split("/")[-1])
    
    # Create an instance of the OpenInterface() to extract URLs
    from biovida.images.openi_interface import OpenInterface
    opi = OpenInterface(cache_path)
    df = opi.image_record_database
    
    # Extract the URL data from the image record database
    df['img_names'] = df['img_cache_path'].map(lambda x: x.split("/")[-1])
    name_url_dict = dict(zip(df['img_names'], df['img_large']))
    
    # Map the URL data onto the image names
    images_used['URL'] = images_used['ImageName'].map(lambda x: name_url_dict[x.replace(" (1)", "")])

    # Save as a csv.
    images_used.to_csv(save_path, index=False)


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

    :param tpl: image shape
    :param border: space from the border of the image
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
arrow_creator(base_images, arrows, n=10000, general_name="arrow", save_location=arrow_save_location)


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
        grid_masher(base_images, name="{0}_{1}".format(i, general_name), save_location=save_location)


# Create the grid training data
# grid_creator(base_images, n=10000, general_name='grid', save_location=grid_save_location)


# ------------------------------------------------------------------------------------------
# Shapes (Ellipses and Square)
# ------------------------------------------------------------------------------------------

# Vary:
#   - the number of ellipses (1-3)
#   - their position and thickness
#   - shape: cicle or oval (about either x or y)
#           - how stretched they are about the elongated axis.


def stretch_generator(stretch_range=(0.4, 0.75)):
    """

    Randomly generates a tuple defining an axis to stretch
    and by how much.

    :param stretch_range:
    :return:
    """
    case = random.choice([0, 1, 2])
    if case == 0: # no stretch
        return 1, 1
    axis_stretch = random.uniform(stretch_range[0], stretch_range[1])
    if case == 1:  # x axis stretch
        return axis_stretch, 0
    elif case == 2:  # y axis stretch
        return 0, axis_stretch


def draw_ellipse(img, position, r, stretch, thickness=4, fill=None, outline='#d3d3d3'):
    """

    :param img:
    :param x:
    :param y:
    :param r:
    :param stretch: tuple of the form (x stretch, y stretch) -- (values must be > 0).
    :param fill:
    :param outline:
    :return:
    """
    # See: http://stackoverflow.com/a/2980931/4898004
    draw = ImageDraw.Draw(img)

    # Extract Stretch (if passed)
    x_s, y_s = tuple(map(int, np.array([r] * 2) * np.array(stretch)))

    # Extract the position
    x, y = position

    split = int(thickness/2)
    for rr in np.arange(r-split, r + split + 1):
        draw.ellipse((x - rr - x_s, y - rr - y_s, x + rr + x_s, y + rr + y_s), outline=outline, fill=fill)
    return img


def ellipse_mash(base_image, border_buffer=0.25, stretch_range=(0.8, 1.2)):
    """

    :param base_image:
    :param border_buffer:
    :param stretch_range:
    :return:
    """
    # Compute the range of radii that will easily fit.
    radius_range = tuple(map(int, (max(base_image.size)*0.025, max(base_image.size)*0.05)))

    for _ in range(randint(1, 2)):
        # Define random properties
        ellipse_position = random_tuple_in_range(base_image.size, border_buffer=border_buffer)
        ellipse_radius = randint(radius_range[0], radius_range[1])
        ellipse_thickness = random.choice([2, 4])
        ellipse_stretch = stretch_generator(stretch_range=stretch_range)

        # Append Image
        base_image = draw_ellipse(base_image, ellipse_position, ellipse_radius, ellipse_stretch, ellipse_thickness)

    return base_image


def ellipse_img_creator(all_img_options, n, general_name, save_location):
    """

    :param all_img_options:
    :param n:
    :param general_name:
    :param save_location:
    :return:
    """
    for i in tqdm(range(1, n+1)):
        # Open a random photo and crop
        base_image = random_crop(Image.open(random.choice(all_img_options)))
        # Save
        ellipse_mash(base_image).save("{0}/{1}_{2}.png".format(save_location, i, general_name))


# ellipse_img_creator(base_images, n=10000, general_name='ellipse', save_location=ellipses_save_location)
























