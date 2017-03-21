"""

    Arrows
    ~~~~~~

"""
import os
import random
import numpy as np

from PIL import Image
from tqdm import tqdm

from data.synthesized_data.support_tools import (avg_color,
                                                 quality,
                                                 resize_image,
                                                 _load_background_min,
                                                 random_stretch,
                                                 random_tuple_away)

from data.synthesized_data._private.my_file_paths import arrow_path, arrow_save_location

# In short, here the desired output is N images with arrows ontop of base images
# These images should not overlap (at least not by very much), be clearly visible
# and appear close to the center of the image


# Random valid MRI as background
# Random arrow
#     - location in image
#     - random size (in some range)
#     - random rotation


# Load arrows
arrows_raw = [i for i in os.listdir(arrow_path) if i.endswith(".png")]
# sorted_arrows = sorted(arrows_raw, key=lambda x: extract_digit(x))
arrows = [os.path.join(arrow_path, i) for i in arrows_raw]


def min_contrast_met(background, foreground, random_background_loc, min_delta=65):
    """

    Compare the foreground and the background to decide if the foreground would be visible if plotted.

    :return:
    """
    # Get the average color about the plot location (0-255)
    avg_background_color_region = avg_color(background, random_background_loc, window=max(foreground.size))

    # Get the median color of the foreground (0-255)
    foreground_color = np.median(np.asarray(foreground))

    # Compute the absolute difference
    if abs(avg_background_color_region - foreground_color) < min_delta:
        return False
    else:
        return True


def arrow_back_foreground_mash(background_options,
                               foreground_options,
                               bg_min_ceiling=295,
                               foreground_scale_range=(0.21, 0.225),
                               foreground_stretch_by=(0.85, 1.25),
                               location_border_buffer=0.38):
    # Load a Background
    background = _load_background_min(background_options, min_size=125)

    # Number of arrows
    prior_positions = list()
    lock_size = random.choice([True, True, False])
    prior_size = None
    attempts = 0
    count = 0
    N = np.random.choice([1, 2, 3, 4, 5], 1, p=[0.48, 0.38, 0.11, 0.02, 0.01])[0]
    while count < N:

        if attempts > 5000:
            background = _load_background_min(background_options, min_size=125)
            prior_size = None
            count = 0
            attempts = 0

        # randomly Load the foreground and convert to grayscale
        forground_choice = random.choice(foreground_options)

        foreground = Image.open(forground_choice).convert("RGBA")

        # Randomly stretch about a given axis
        foreground = random_stretch(foreground, foreground_stretch_by)

        # Change contrast
        # foreground = foreground.point(lambda p: p * random.uniform(0.87, 0.92))

        if (lock_size and prior_size is None) or lock_size is False:
            # Scale the foreground by a random amount in some range
            foreground_scalar = random.uniform(foreground_scale_range[0], foreground_scale_range[1])
        elif lock_size and prior_size is not None:
            foreground_scalar = prior_size

        # Compute the correct scalar to make foreground `foreground_scalar` times
        # the size of the smallest axis in the background.
        bg_min_axis = min(background.size) if min(background.size) < bg_min_ceiling else bg_min_ceiling
        scale_image_by = (min(background.size) * (foreground_scalar)) / max(foreground.size)

        # Rescale the foreground and rotate; expand=1 expands the canvis to stop the image being cut off
        foreground = resize_image(foreground, scale_image_by).rotate(random.randint(0, 360), expand=1)

        # Random location to place the image
        random_background_loc = random_tuple_away(background.size, prior_positions, location_border_buffer)
        prior_positions.append(random_background_loc)

        # Paste the foreground onto the background IF it would be visible
        if min_contrast_met(background, foreground, random_background_loc):
            background.paste(foreground, random_background_loc, foreground)
            count += 1

        attempts += 1

    return background


def arrow_masher(background_options, foreground_options, name, save_location):
    """

    :param background_options:
    :param foreground_options:
    :return:
    """
    # Compute mash and convert to grayscale
    rslt = arrow_back_foreground_mash(background_options, foreground_options).convert("LA")

    # Save to disk
    rslt.save("{0}/{1}.png".format(save_location, name), quality=quality)


def arrow_creator(background_options, foreground_options, start, end, general_name, save_location):
    """

    :param background_options:
    :param foreground_options:
    :param n:
    :param save_location:
    :return:
    """
    for i in tqdm(range(start+1, end+1)):
        arrow_masher(background_options, foreground_options, name="{0}_{1}".format(i, general_name), save_location=save_location)


# Create the arrow synthesized_data data
# arrow_creator(base_images, arrows, 17000, 30000, general_name="arrow", save_location=arrow_save_location)

