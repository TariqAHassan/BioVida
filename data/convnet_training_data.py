"""

    Create Dataset for CNN Image Problem Classification
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
from os.path import join as os_join
from biovida.images import ImageProcessing
from biovida.images.image_cache_mgmt import image_divvy

from data.source_data.training_images_from_openi import opi

from data.synthesized_data.support_tools import base_images
from data.synthesized_data.text import occluding_text_creator


from data._private.paths import output_dir


# ----------------------------------------------------------------------------------------------------------
# Source Data
# ----------------------------------------------------------------------------------------------------------


ip = ImageProcessing(opi, db_to_extract='cache_records_db')
ip.grayscale_analysis()


def my_divvy_rule(row):
    if row['grayscale'] == True:
        if 'grids' == row['image_problems_from_text'][0]:
            return os_join(output_dir, 'grids')
        elif 'arrows' == row['image_problems_from_text']:
            return  os_join(output_dir, 'arrows')

divvy_info = image_divvy(ip, divvy_rule=my_divvy_rule, action='copy')


# ----------------------------------------------------------------------------------------------------------
# Synthesized Data
# ----------------------------------------------------------------------------------------------------------


occluding_text_creator(base_images, start=0, end=20000, general_name="text", save_location=os_join(output_dir, "text"))


























































