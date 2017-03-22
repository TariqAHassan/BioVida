"""

    Valid Images
    ~~~~~~~~~~~~

"""
import os
import random
from PIL import Image
from tqdm import tqdm

from data.synthesized_data.support_tools import (base_images,
                                                 resize_image,
                                                 random_crop,
                                                 quality)

from data.synthesized_data._private.my_file_paths import valid_image_save_location


def valid_image_creator(image_options, start, end, general_name, save_location):
    """

    """
    for i in tqdm(range(start+1, end+1)):
        # Open and randomly crop
        image = random_crop(Image.open(random.choice(image_options)))
        # Randomly rescale
        image = resize_image(image, random.uniform(0.8, 1.2))
        # Save
        image.save(os.path.join(save_location, "{0}_{1}.png".format(i, general_name)), quality=quality)
