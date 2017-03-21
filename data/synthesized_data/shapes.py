"""

    Shapes (Ellipses and Square)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import random
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageDraw
from random import randint

from data.synthesized_data.support_tools import (base_images,
                                                 quality,
                                                 random_crop,
                                                 random_tuple_in_range)


from data.synthesized_data._private.my_file_paths import ellipses_save_location


# Vary:
#   - the number of ellipses (1-3)
#   - their position and thickness
#   - shape: circle or oval (about either x or y)
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


def draw_ellipse(image, position, r, stretch, thickness=4, fill=None, outline='#d3d3d3'):
    """

    :param image:
    :param position:
    :param r:
    :param stretch: tuple of the form (x stretch, y stretch) -- (values must be > 0).
    :param thickness:
    :param fill:
    :param outline:
    :return:
    """
    # See: http://stackoverflow.com/a/2980931/4898004
    draw = ImageDraw.Draw(image)

    # Extract Stretch (if passed)
    x_s, y_s = tuple(map(int, np.array([r] * 2) * np.array(stretch)))

    # Extract the position
    x, y = position

    split = int(thickness/2)
    for rr in np.arange(r-split, r + split + 1):
        draw.ellipse((x - rr - x_s, y - rr - y_s, x + rr + x_s, y + rr + y_s), outline=outline, fill=fill)
    return image


def ellipse_mash(base_image, border_buffer=0.25, stretch_range=(0.8, 1.2)):
    """

    :param base_image:
    :param border_buffer:
    :param stretch_range:
    :return:
    """
    # Compute the range of radii that will easily fit.
    radius_range = tuple(map(int, (max(base_image.size) * 0.025, max(base_image.size) * 0.05)))

    for _ in range(randint(1, 2)):
        # Define random properties
        ellipse_position = random_tuple_in_range(base_image.size, border_buffer=border_buffer)
        ellipse_radius = randint(radius_range[0], radius_range[1])
        ellipse_thickness = random.choice([2, 4])
        ellipse_stretch = stretch_generator(stretch_range=stretch_range)

        # Append Image
        base_image = draw_ellipse(base_image, ellipse_position, ellipse_radius, ellipse_stretch, ellipse_thickness)

    return base_image


def ellipse_image_creator(all_image_options, start, end, general_name, save_location):
    """

    :param all_image_options:
    :param n:
    :param general_name:
    :param save_location:
    :return:
    """
    for i in tqdm(range(start+1, end+1)):
        # Open a random photo and crop
        base_image = random_crop(Image.open(random.choice(all_image_options)))
        # Save
        ellipse_mash(base_image).save("{0}/{1}_{2}.png".format(save_location, i, general_name), quality=quality)


ellipse_image_creator(base_images, start=0, end=30000, general_name='ellipse', save_location=ellipses_save_location)

