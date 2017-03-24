"""

    Valid Images
    ~~~~~~~~~~~~

"""
import os
import random
from tqdm import tqdm

from data.synthesized_data.support_tools import (resize_image,
                                                 random_crop_min,
                                                 QUALITY,
                                                 MIN_SIZE)


def valid_image_creator(image_options, start, end, general_name, save_location, min_size=MIN_SIZE):
    """

    """
    scale_range = (0.8, 1.2)
    required_min_size = min_size * (2 - scale_range[0])  # simple algebra

    for i in tqdm(range(start+1, end+1)):
        # Open and randomly crop
        image = random_crop_min(background_options=image_options, min_size=required_min_size, limit=500)
        # Randomly rescale
        image = resize_image(image, random.uniform(scale_range[0], scale_range[1]))

        # Save
        save_path = os.path.join(save_location, "{0}_{1}.png".format(i, general_name))
        image.save(save_path, quality=QUALITY)
