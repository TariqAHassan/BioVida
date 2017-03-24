"""

    Create Dataset for CNN Image Problem Classification
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import os
from os.path import join as os_join
from biovida.images import OpeniInterface
from biovida.images import ImageProcessing
from biovida.images.image_cache_mgmt import image_divvy
from biovida.support_tools.utilities import train_val_test

from data.synthesized_data.support_tools import base_images
from data.synthesized_data.text import occluding_text_creator
from data.synthesized_data.arrows import arrows, arrow_creator
from data.synthesized_data.valid import valid_image_creator
from data.synthesized_data.grids import grid_creator

from data._private.paths import output_dir
from data._private.paths import target_dir


# ----------------------------------------------------------------------------------------------------------
# Source Data
# ----------------------------------------------------------------------------------------------------------

opi = OpeniInterface()

def divvy_rule(row):
    if 'grids' == row['image_problems_from_text'][0]:
        return os_join(output_dir, 'grids')
    elif 'arrows' == row['image_problems_from_text'][0]:
        return os_join(output_dir, 'arrows')


divvy_info = image_divvy(opi, db_to_extract='cache_records_db',
                         divvy_rule=divvy_rule, action='copy')


# ----------------------------------------------------------------------------------------------------------
# Synthesized Data (Either Entirely or to Compliment Source Data)
# ----------------------------------------------------------------------------------------------------------


TOTAL_PER_GROUP = 30000


def number_of_images_in_dir(path):
    return len([i for i in os.listdir(path) if i.endswith(".png")])


# -----------------------------------------
# Valid
# -----------------------------------------


# os.makedirs(os_join(output_dir, "valid_image"))
valid_image_creator(base_images, start=0, end=TOTAL_PER_GROUP,
                    general_name="valid_image", save_location=os_join(output_dir, "valid_image"))


# -----------------------------------------
# Text
# -----------------------------------------


# os.makedirs(os_join(output_dir, "text"))
occluding_text_creator(base_images, start=0, end=TOTAL_PER_GROUP,
                       general_name="text", save_location=os_join(output_dir, "text"))


# -----------------------------------------
# Arrows
# -----------------------------------------


arrows_end = TOTAL_PER_GROUP - number_of_images_in_dir(os_join(output_dir, 'arrows'))
# 17287

if arrows_end > 0:
    arrow_creator(base_images, arrows, start=0, end=arrows_end,
                  general_name="arrow", save_location=os_join(output_dir, 'arrows'))
    

# -----------------------------------------
# Grids
# -----------------------------------------


grids_end = TOTAL_PER_GROUP - number_of_images_in_dir(os_join(output_dir, 'grids'))
# 12905

if grids_end > 0:
    grid_creator(base_images, start=0, end=grids_end,
                 general_name='grid', save_location=os_join(output_dir, 'grids'))


# ----------------------------------------------------------------------------------------------------------
# Split into Train/Val/Test
# ----------------------------------------------------------------------------------------------------------


output = train_val_test(data=output_dir, train=0.75, validation=0.15, test=0.1,
                        action='copy', target_dir=target_dir)




















