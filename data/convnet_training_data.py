"""

    Create Dataset for CNN Image Problem Classification
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import os
from os.path import join as os_join
from biovida.images import OpeniInterface
from biovida.images import OpeniImageProcessing
from biovida.images import image_delete

from biovida.images.image_cache_mgmt import image_divvy
from biovida.support_tools.utilities import train_val_test

from data.synthesized_data.support_tools import base_images
from data.synthesized_data.text import occluding_text_creator
from data.synthesized_data.arrows import arrow_creator
from data.synthesized_data.valid import valid_image_creator
from data.synthesized_data.grids import grid_creator

from data._private.paths import output_dir
from data._private.paths import target_dir


# ----------------------------------------------------------------------------------------------------------
# Source: Harvesting
# ----------------------------------------------------------------------------------------------------------


# Note: this entails downloading over 250,000 records (will take some time)

opi = OpeniInterface()
opi.search(image_type=['mri', 'ct', 'x_ray'])
df = opi.pull(download_limit=None, image_size=None, clinical_cases_only=False)

opi.records_db = opi.records_db[
    opi.records_db['image_problems_from_text'].map(lambda x: isinstance(x, tuple) and len(x) == 1)].reset_index(drop=True)

# The restriction above resulted in ~81,000 records,
# i.e., 81k images were downloaded.

opi.pull(new_records_pull=False)


# ----------------------------------------------------------------------------------------------------------
# Source: Delete Images Which are Not Grayscale
# ----------------------------------------------------------------------------------------------------------


ip = OpeniImageProcessing(opi, db_to_extract='records_db')
ip.grayscale_analysis()


def delete_rule(row):
    if row['grayscale'] != True:
        return True


image_delete(ip, delete_rule=delete_rule)
# Resulted in ~30,000 images


# ----------------------------------------------------------------------------------------------------------
# Source: Divvy
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
# Synthesized Data: Generate (Either Entirely or to Compliment Source Data)
# ----------------------------------------------------------------------------------------------------------


START = 0
TOTAL_PER_GROUP = 75000


def number_of_images_in_dir(path):
    return len([i for i in os.listdir(path) if i.endswith(".png")])


# -----------------------------------------
# Valid
# -----------------------------------------


os.makedirs(os_join(output_dir, "valid_image"))
valid_image_creator(base_images, start=START, end=TOTAL_PER_GROUP,
                    general_name="valid_image", save_location=os_join(output_dir, "valid_image"))


# -----------------------------------------
# Text
# -----------------------------------------


os.makedirs(os_join(output_dir, "text"))
occluding_text_creator(base_images, start=START, end=TOTAL_PER_GROUP,
                       general_name="text", save_location=os_join(output_dir, "text"))


# -----------------------------------------
# Arrows
# -----------------------------------------


arrows_end = TOTAL_PER_GROUP - number_of_images_in_dir(os_join(output_dir, 'arrows'))
# 62,287

if arrows_end > 0:
    arrow_creator(base_images, start=START, end=arrows_end,
                  general_name="arrow", save_location=os_join(output_dir, 'arrows'))


# -----------------------------------------
# Grids
# -----------------------------------------


grids_end = TOTAL_PER_GROUP - number_of_images_in_dir(os_join(output_dir, 'grids'))
# 57,905

if grids_end > 0:
    grid_creator(base_images, start=START, end=grids_end,
                 general_name='grid', save_location=os_join(output_dir, 'grids'))


# ----------------------------------------------------------------------------------------------------------
# Split into Train/Val/Test
# ----------------------------------------------------------------------------------------------------------


output = train_val_test(data=output_dir, train=0.75, validation=0.15, test=0.1,
                        action='move', target_dir=target_dir, delete_source=True)
