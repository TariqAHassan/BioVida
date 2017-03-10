"""

    Tools to Create Training Data for Image Recognition
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import PIL
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import StringIO
from PIL import ImageDraw
from PIL import ImageFont
from random import randint
from itertools import chain

from data.training.my_file_paths import (cache_path,
                                         arrow_path,
                                         base_image_path,
                                         grid_save_location,
                                         occluding_text_save_location,
                                         arrow_save_location,
                                         valid_image_save_location,
                                         ellipses_save_location)


# ToDo: create 'grids' with black spaces added to each image on either side images.
quality = 95  # set image quality


# Note: Some of this code is somewhat messy because I am still prototyping how to solve these problems.


# ------------------------------------------------------------------------------------------
# Define Images to use
# ------------------------------------------------------------------------------------------


# Define Images to Use
base_images = [os.path.join(base_image_path, i) for i in os.listdir(base_image_path) if i.endswith(".png")]


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


def open_muliple_and_random_crop(image_list):
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


# ------------------------------------------------------------------------------------------
# valid_image
# ------------------------------------------------------------------------------------------


def valid_image_creator(image_options, start, end, general_name, save_location):
    """

    :param image_options:
    :param n:
    :param general_name:
    :param save_location:
    :return:
    """
    for i in tqdm(range(start+1, end+1)):
        # Open and randomly crop
        image = random_crop(Image.open(random.choice(image_options)))
        # Randomly rescale
        image = resize_image(image, random.uniform(0.8, 1.2))
        # Save
        image.save(os.path.join(save_location, "{0}_{1}.png".format(i, general_name)), quality=quality)


# valid_image_creator(base_images, 17000, 30000, "valid_image", valid_image_save_location)

# ------------------------------------------------------------------------------------------
# Arrows
# ------------------------------------------------------------------------------------------

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


def distance_between_points(point_a, point_b):
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


def _load_background(background_options):
    """

    :param background_options:
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


def _load_background_min(background_options, min_size, limit=250):
    """

    :param background_options:
    :param min_size:
    :param limit:
    :return:
    """
    bg = _load_background(background_options)

    c = 0
    while c <= limit and min(bg.size) < min_size:
        bg =  _load_background(background_options)
        c += 1

    return bg


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


def arrow_back_foreground_mash(background_options
                               , foreground_options
                               , bg_min_ceiling=295
                               , foreground_scale_range=(0.21, 0.225)
                               , foreground_stretch_by=(0.85, 1.25)
                               , location_border_buffer=0.38):
    """

    :param background_path:
    :param foreground_path:
    :param foreground_scalar:
    :param location_border_buffer:
    :return:
    """
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
            foreground_sclar = prior_size

        # Compute the correct scalar to make foreground `foreground_scalar` times
        # the size of the smallest axis in the background.
        bg_min_axis = min(background.size) if min(background.size) < bg_min_ceiling else bg_min_ceiling
        scale_image_by = (min(background.size) * (foreground_scalar)) / max(foreground.size)

        # Rescale the foreground and rotate; expand=1 expands the canvis to stop the image being cut off
        foreground = resize_image(foreground, scale_image_by).rotate(randint(0, 360), expand=1)

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


# Create the arrow training data
# arrow_creator(base_images, arrows, 17000, 30000, general_name="arrow", save_location=arrow_save_location)


# ------------------------------------------------------------------------------------------
# Grids
# ------------------------------------------------------------------------------------------


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

    :param image_list:
    :return:
    """
    # see: http://stackoverflow.com/a/30228789/4898004.
    if min_shape_override is not None:
        images = image_list
        min_shape = min_shape_override
    else:
        images = open_muliple_and_random_crop(image_list)
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
    images = open_muliple_and_random_crop(image_list)

    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]

    # Generate the first grid
    grid_1 = image_stacker(image_list=images[:int(len(images)/2)], stacker=stacker_a, min_shape_override=min_shape)

    # Generate the second grid
    grid_2 = image_stacker(image_list=images[int(len(images)/2):], stacker=stacker_a, min_shape_override=min_shape)

    # Stack side-by-side
    return stacker_b((grid_1, grid_2))


def grid_masher(all_image_options, name, save_location):
    """

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

    # Image.fromarray(grid).show()
    rslt = Image.fromarray(grid)

    # Convert to an image and Save to disk
    rslt.save("{0}/{1}.png".format(save_location, name), quality=quality)


def grid_creator(all_image_options, start, end, general_name, save_location):
    """

    :param all_image_options:
    :param n:
    :param save_location:
    :return:
    """
    for i in tqdm(range(start+1, end+1)):
        grid_masher(base_images, name="{0}_{1}".format(i, general_name), save_location=save_location)


# Create the grid training data
# grid_creator(base_images, 17000, 35000, general_name='grid', save_location=grid_save_location)


# ------------------------------------------------------------------------------------------
# Text
# ------------------------------------------------------------------------------------------


def get_fonts():

    font_locations = ["/Library/Fonts", "/System/Library/Fonts/"] # This is only valid on macOS!
    def font_extract(font_path, allowed_formats=('.ttf', 'ttc')):
        def allowed_tails(s):
            return any(s.endswith(a) for a in allowed_formats)
        return [os.path.join(font_path, font) for font in os.listdir(font_path) if allowed_tails(font)]

    font_complete_list = list(chain(*map(font_extract, font_locations)))
    allowed_font_famlies = ('arial', 'courier', 'times', 'sans ')
    banned_fonts = ['unicode', 'arialhb']

    # Define a list of allowed fonts.
    all_fonts = [f for f in font_complete_list if any(i in f.lower() for i in allowed_font_famlies)
                 and not any(i in f.lower() for i in banned_fonts)]

    return all_fonts

all_fonts = get_fonts()

# Source: https://www.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html
letter_data = """Letter,Count,Letter,Frequency
E,21912,E,12.02
T,16587,T,9.10
A,14810,A,8.12
O,14003,O,7.68
I,13318,I,7.31
N,12666,N,6.95
S,11450,S,6.28
R,10977,R,6.02
H,10795,H,5.92
D,7874,D,4.32
L,7253,L,3.98
U,5246,U,2.88
C,4943,C,2.71
M,4761,M,2.61
F,4200,F,2.30
Y,3853,Y,2.11
W,3819,W,2.09
G,3693,G,2.03
P,3316,P,1.82
B,2715,B,1.49
V,2019,V,1.11
K,1257,K,0.69
X,315,X,0.17
Q,205,Q,0.11
J,188,J,0.10
Z,128,Z,0.07
"""

def letters_freqs():
    """

    Define lists of letter Freqs.

    :return:
    """
    letter_freq = pd.read_csv(StringIO(letter_data), sep=",")[['Letter', 'Count']]
    letter_freq_dict = dict(zip(letter_freq['Letter'], letter_freq['Count']))
    letter_freq_dict = {k.lower(): float(v / sum(letter_freq_dict.values())) for k, v in letter_freq_dict.items()}

    # Procedual approach (not very fancy, but reduces the chance of any mix ups).
    letters, freqs = list(), list()
    for k, v in letter_freq_dict.items():
        letters.append(k)
        freqs.append(v)

    return letters, freqs


# Drop into globals
letters, freqs = letters_freqs()


def random_text_gen(n_chars=(1, 3), n_words=(2, 3), n_phrases=(1, 2)):
    """

    :param n:
    :return:
    """
    def random_int(tpl):
        return randint(tpl[0], tpl[1])

    def random_letters():
        return np.random.choice(letters, random_int(n_chars), p=freqs)

    def random_numbers():
        numbers = list(np.random.choice(list(map(str, range(1, 11))), random_int(n_chars)))
        def random_astrix(n):
            options = [("*", n, ""), ("", n, "*"), ("", n, "")]
            return "".join(options[randint(0, 2)])
        return [random_astrix(n) if randint(0, 3) == 1 else n for n in numbers]

    def pseudo_words():
        return ("".join(random_letters()) for _ in range(random_int(n_words)))

    # Define a container for the phrases
    phrases = list()

    # Randomly decide to add 1-2 asterisks, or none.
    phrases += ["*"] * np.random.choice([0, 1, 2], 1, p=[0.90, 0.09, 0.01])[0]

    # Decide whether or not to only return numbers (and possibly asterisks)
    if randint(0, 1) == 1:
        return random_numbers() + phrases

    # Create a random number of pseudo phrases
    phrases += [" ".join(pseudo_words()) for _ in range(random_int(n_phrases))]

    # Randomly decide to add some natural numbers
    if randint(0, 1) == 1:
        phrases += random_numbers()

    # Return words composed of single letters as seperate elements
    return list(chain(*[i.split() if max(map(len, i.split())) == 1 else [i] for i in phrases]))


def font_size_picker(image, text, font_name, text_size_proportion):
    """

    :param image:
    :param text:
    :param font_name:
    :param text_size_proportion:
    :return:
    """

    def font_size(text, font_name, size_of_font):
        f = ImageFont.truetype(font_name, size_of_font)
        return f.getsize(text)

    # Get the width of the image
    image_width, _ = image.size

    # Compute the ideal width of the text
    goal_size = text_size_proportion * image_width

    # Get Width of the text all all font sizes from 1 to 100
    all_sizes = (font_size(text, font_name, int(s))[0] for s in range(1, 80))

    # Find the font size which is closest to the desired width
    return min(all_sizes, key=lambda x: abs(x - goal_size))


def add_text(image, text, font, position, color):
    """

    :param image:
    :param text:
    :param font:
    :param position:
    :param size:
    :param color:
    :return:
    """
    draw = ImageDraw.Draw(image)
    draw.text(position, text, color, font=font)
    return image


def text_background_window(background, text_loc, text_dim, black_color_threshold=35):
    """

    Summary of a the background for a text block.
    Currently computes the median color for this region.

    :param background:
    :param text_loc:
    :param text_dim:
    :return:
    """
    # Crop = (left, upper, right, lower)
    # Get the bounds
    right = min(background.size[0], text_loc[0] + text_dim[0])
    lower = min(background.size[1], text_loc[1] + text_dim[1])

    # Select the corresponding part of the background
    cropped_background = background.crop((text_loc[0], text_loc[1], right, lower))

    # Background as an array
    background_arr = np.asarray(cropped_background)

    # Test if the background is all black
    all_black_background = all(i < black_color_threshold for i in background_arr.flatten())

    # Return the median color
    return np.mean(background_arr), all_black_background


# ---------------------------------------------
# Occluding Text
# ---------------------------------------------


def occluding_text_masher(background_options, border_buffer=0.40):
    """

    Generate Images where the text occludes the image.
    The probability of this function working are tied to `border_buffer`.
    The higher this value, the more likely the images will actually occlude the image.

    :param background_options:
    :return:
    """
    # Generate Some random text
    list_of_text = random_text_gen(n_words=(2, 2))

    # Randomly load and crop a background
    background = random_crop(_load_background_min(background_options, min_size=150))

    c, attempts = 0, 0
    prior_locations = list()
    while c < len(list_of_text):

        text = list_of_text[c]

        # Reset if it appears an impossible task has been created.
        if attempts > 500:
            c, attempts = 0, 0
            prior_locations = list()
            background = random_crop(_load_background_min(background_options, min_size=150))

        # 1. Get a random location on the image
        text_loc = random_tuple_away(background.size, prior_locations, border_buffer=border_buffer)
        prior_locations.append(text_loc)

        # 2. Randomly pick a size for the font
        text_size_proportion = np.random.uniform(0.055, 0.07)

        # 3. Randomly pick a font
        font_name = np.random.choice(all_fonts)

        # 4. Compute the size of the font required span the desired proportion of the image
        size_of_font = font_size_picker(background, text, font_name, text_size_proportion)

        # 5. Generate the font
        font = ImageFont.truetype(font_name, size_of_font)

        # 6. Get the size of the font
        text_dim = font.getsize(text)

        # 7. Get the color of the background in that area
        summary_background_color = text_background_window(background, text_loc, text_dim)

        # Check the background is 1. on average not black and 2. has some variance (sd).
        if not summary_background_color[1]: # ToDo: THis is not working quite right...
            # 8. Choose the text color
            text_color = opposite_color(summary_background_color[0])

            # 9. Add the text
            background = add_text(background, text, font, text_loc, text_color)

            # Update record of text that has been appended to the background
            c += 1

        attempts += 1

    return background


def occluding_text_creator(all_image_options, start, end, general_name, save_location):
    """

    :param all_image_options:
    :param n:
    :param save_location:
    :return:
    """
    for i in tqdm(range(start+1, end+1)):
        # Define the save location
        image_name ="{0}_{1}.png".format(i, general_name)
        save_path = os.path.join(save_location, image_name)
        # Generate and Save
        occluding_text_masher(all_image_options).save(save_path)


occluding_text_creator(base_images, 0, 33000, "occluding_text", occluding_text_save_location)


# ToDo: Non-Occluding Text


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


def draw_ellipse(image, position, r, stretch, thickness=4, fill=None, outline='#d3d3d3'):
    """

    :param image:
    :param x:
    :param y:
    :param r:
    :param stretch: tuple of the form (x stretch, y stretch) -- (values must be > 0).
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


# ellipse_image_creator(base_images, start=0, end=30000, general_name='ellipse', save_location=ellipses_save_location)



































