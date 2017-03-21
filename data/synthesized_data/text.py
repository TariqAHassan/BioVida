"""

    Text
    ~~~~

"""
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from io import StringIO
from PIL import ImageDraw
from PIL import ImageFont
from random import randint
from itertools import chain

from data.synthesized_data.support_tools import (base_images,
                                                 random_tuple_away,
                                                 opposite_color,
                                                 random_crop,
                                                 _load_background_min)


from data.synthesized_data._private.my_file_paths import occluding_text_save_location


def get_fonts():

    font_locations = ["/Library/Fonts", "/System/Library/Fonts/"]  # This is only valid on macOS!

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

    # Return words composed of single letters as separate elements
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
