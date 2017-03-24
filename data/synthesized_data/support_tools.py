"""

    Tools to Create Training Data for Image Recognition
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from random import randint

from data.synthesized_data._private.my_file_paths import valid_x_ray
from data.synthesized_data._private.my_file_paths import valid_mri_ct

# ToDo: create 'grids' with black spaces added to each image on either side images.
QUALITY = 95   # set image quality
MIN_SIZE = 150  # the smallest an image can be w.r.t. a single axis (e.g., 140 x 300 is not allowed)


np.random.seed(100)  # make dataset reproducible


# ----------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------


def _full_paths(path):
    return [os.path.join(path, i) for i in os.listdir(path) if i.endswith(".png")]


base_images = _full_paths(valid_mri_ct) + _full_paths(valid_x_ray)


# ----------------------------------------------------------------------------------------------------------
# Define Images to use
# ----------------------------------------------------------------------------------------------------------


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
    from biovida.images.openi_interface import OpeniInterface
    opi = OpeniInterface(cache_path)
    df = opi.cache_records_db
    
    # Extract the URL data from the image record database
    df['image_names'] = df['image_cache_path'].map(lambda x: x.split("/")[-1])
    name_url_dict = dict(zip(df['image_names'], df['image_large']))
    
    # Map the URL data onto the image names
    images_used['URL'] = images_used['ImageName'].map(lambda x: name_url_dict[x.replace(" (1)", "")])

    # Save as a csv.
    images_used.to_csv(save_path, index=False)


# ------------------------------------------------------------------------------------------
# General Tools
# ------------------------------------------------------------------------------------------


def avg_color(background, location, window=50):
    """

    Gets the average color about some location
    Note: this may have bugs!

    :param background: PIL image
    :param location: (w, h)
    :type location: tuple
    :param window: region
    :type window: ``int``
    :return:
    :rtype: ``float``
    """

    # Get the size of the background
    bw, bh = background.size

    # Get the location the arrow would be plot
    rw, rh = location

    # Compute the window which is actually possible
    left, upper = max(0, rw - window), max(0, rh - window)  # kind of looks like Relu!
    right, lower = min(bw, rw + window), min(bh, rh + window)

    # Crop
    bg_cropped = background.crop((left, upper, right, lower))

    # Report the average RGB color.
    return np.mean(np.asarray(bg_cropped))


def resize_image(image, scalar=1/3, min_size=15):
    """

    :param image:
    :param scalar:
    :param min_size: the min. size in pixels that a resized image can be;
                     pass 0 to disable.
    :return:
    """
    width, height = image.size

    new_width, new_height = int(width * scalar), int(height * scalar)

    if new_width < min_size or new_height < min_size:
        new_width += abs(new_width - min_size)
        new_height += abs(new_height - min_size)

    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return image


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


def random_crop(image, choice_override=None, top_crop=0.2, lower_crop=0.9):
    """

    :param image:
    :param choice_override:
    :param top_crop: the amount of the top of the image to crop
    :param lower_crop: the amount of the botto of the image to crop
    :return:
    """
    w, h = image.size
    if top_crop is not None:
        image = image.crop((0, int(h * top_crop), w, h))
    w, h = image.size
    if lower_crop is not None:
        image = image.crop((0, 0, w, int(h * lower_crop)))

    w, h = image.size
    w2, h2 = np.round(np.array(image.size) / 2).astype(int)
    if isinstance(choice_override, int):
        choice = choice_override
    else:
        choice = randint(0, 2)

    side = randint(0, 1)
    if choice == 0:
        return image
    elif choice == 1: # crop w.r.t. width
        if side == 0:
            return image.crop((0, 0, w2, h))
        elif side == 1:
            return image.crop((w2, 0, w, h))
    elif choice == 2: # crop w.r.t. height
        if side == 0:
            return image.crop((0, 0, w, h2))
        elif side == 1:
            return image.crop((0, h2, w, h))


def random_stretch(image, stretch_by):
    """

    :param image:
    :param stretch_by:
    :return:
    """
    w, h = image.size
    scale_by = random.uniform(stretch_by[0], stretch_by[1])
    choice = randint(0, 2)

    if choice == 0:
        return image
    elif choice == 1:
        return image.resize((int(w * scale_by), h), Image.ANTIALIAS)
    elif choice == 2:
        return image.resize((w, int(h * scale_by)), Image.ANTIALIAS)


def open_multiple_and_random_crop(image_list):
    """

    :param image_list:
    :return:
    """
    to_crop_or_not_to_crop = randint(0, 1)
    if to_crop_or_not_to_crop == 0:
        return [random_crop(Image.open(i), choice_override=0) for i in image_list]
    else:
        return [random_crop(Image.open(i), choice_override=randint(1, 2)) for i in image_list]


def opposite_color(color_val, buffer=7.5):
    """

    :param color_val:
    :param buffer:
    :return:
    """
    color_midpoint = 255 / 2.0

    upper = color_midpoint + buffer
    lower = color_midpoint - buffer

    if color_val > upper:
        color = randint(0, 10)
    elif color_val < lower:
        color = randint(245, 255)
    else:
        color = randint(int(lower), int(np.ceil(upper)))

    # Return a pure color
    return tuple([color] * 3)


def load_background(background_options):
    """

    :param background_options:
    :type background_options: ``list``
    :return:
    """
    # Select random element
    background_path = random.choice(background_options)

    # Load a Background
    background = Image.open(background_path).convert("RGBA")

    # Randomly crop background
    background = random_crop(background)

    # Randomly rescale background
    background = resize_image(background, random.uniform(0.8, 1.2))

    return background


def load_background_min(background_options, min_size, limit=500):
    """

    :param background_options:
    :param min_size:
    :param limit:
    :return:
    """
    bg = load_background(background_options)

    c = 0
    while c <= limit and min(bg.size) < min_size:
        bg = load_background(background_options)
        c += 1

    return bg


def random_crop_min(background_options, min_size, limit=250):
    """

    Returns a cropped image of a min. size.

    :param background_options:
    :param min_size:
    :param limit:
    :return:
    :rtype: ``PIL``
    """
    cropped = random_crop(load_background(background_options))

    c = 0
    while c <= limit and min(cropped.size) < min_size:
        cropped = random_crop(load_background(background_options))
        c += 1

    return cropped


def distance_between_points(point_a, point_b):
    """

    Vectorized Euclidean Distance

    :param point_a:
    :param point_b:
    :return:
    """
    return np.sqrt(np.sum((np.array(point_a) - np.array(point_b))**2))


def distance_all(new_position, prior_locations, min_distance_appart):
    l = [distance_between_points(new_position, i) for i in prior_locations]
    return all(i >= min_distance_appart for i in l), min(l)


def random_tuple_away(background_shape, prior_locations, border_buffer, min_sep=0.3, limt=350):
    """

    Compute a location that is some min. distance away from all prior locations

    :param background_shape:
    :param prior_locations:
    :param border_buffer:
    :param min_sep:
    :return:
    """
    # Compute a new location
    new_position = random_tuple_in_range(background_shape, border_buffer)

    # Return if no prior locations
    if not len(prior_locations):
        return new_position

    # Compute the distance that arrows must be appart
    min_distance_appart = int(min(background_shape) * min_sep)

    # Compute distances
    d = distance_all(new_position, prior_locations, min_distance_appart)

    best = (new_position, d[1])

    c = 0
    while not d[0] and c <= limt:
        new_position = random_tuple_in_range(background_shape, border_buffer)
        d = distance_all(new_position, prior_locations, min_distance_appart)

        if d[0]:
            return new_position
        if d[1] > best[1]:
            best = (new_position, d[1])
        c += 1

    return best[0]
