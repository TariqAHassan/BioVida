"""

    Create Dataset for CNN Image Problem Classification
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import os
from os.path import join as os_join
from biovida.images import ImageProcessing
from biovida.images.image_cache_mgmt import image_divvy

from data.source_data.training_images_from_openi import opi

from data.synthesized_data.support_tools import base_images

from data.synthesized_data.text import occluding_text_creator
from data.synthesized_data.arrows import arrows, arrow_creator
from data.synthesized_data.grids import grid_creator


from data._private.paths import output_dir


# ----------------------------------------------------------------------------------------------------------
# Source Data
# ----------------------------------------------------------------------------------------------------------


ip = ImageProcessing(opi, db_to_extract='cache_records_db')
ip.grayscale_analysis()


def divvy_rule(row):
    if row['grayscale'] == True:
        if 'grids' == row['image_problems_from_text'][0]:
            return os_join(output_dir, 'grids')
        elif 'arrows' == row['image_problems_from_text'][0]:
            return os_join(output_dir, 'arrows')


divvy_info = image_divvy(ip, divvy_rule=divvy_rule, action='copy')


# ----------------------------------------------------------------------------------------------------------
# Synthesized Data (Either Entirely or to Compliment Source Data)
# ----------------------------------------------------------------------------------------------------------


TOTAL_PER_GROUP = 30000


# -----------------------------------------
# Text
# -----------------------------------------


os.makedirs(os_join(output_dir, "text"))
occluding_text_creator(base_images, start=0, end=TOTAL_PER_GROUP,
                       general_name="text", save_location=os_join(output_dir, "text"))


# -----------------------------------------
# Arrows
# -----------------------------------------


arrows_end = TOTAL_PER_GROUP - len(os.listdir(os_join(output_dir, 'arrows')))

if arrows_end > 0:
    arrow_creator(base_images, arrows, start=0, end=arrows_end,
                  general_name="arrow", save_location=os_join(output_dir, 'arrows'))
    

# -----------------------------------------
# Grids
# -----------------------------------------


grids_end = TOTAL_PER_GROUP - len(os.listdir(os_join(output_dir, 'grids')))

if grids_end > 0:
    grid_creator(base_images, start=0, end=grids_end,
                 general_name='grid', save_location=os_join(output_dir, 'grids'))





























