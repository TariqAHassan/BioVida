"""

    Valid Images
    ~~~~~~~~~~~~

"""
# Imports
import pandas as pd
from tqdm import tqdm
from biovida.images import image_delete
from biovida.images import OpeniInterface
from biovida.images import ImageProcessing
from data._private.paths import output_dir


# ----------------------------------------------------------------------------------------------------------
# MRI/CT
# ----------------------------------------------------------------------------------------------------------


opi = OpeniInterface()
opi.search(image_type=['mri', 'ct'])
opi.pull(download_limit=20000, image_size=None)

opi.records_db = opi.records_db[pd.isnull(opi.records_db['image_problems_from_text'])].reset_index(drop=True)

opi.pull(new_records_pull=False, download_limit=None)


# ----------------------------------------------------------------------------------------------------------
# X-rays
# ----------------------------------------------------------------------------------------------------------


opi = OpeniInterface()
opi.search(image_type=['x_ray'])
opi.pull(download_limit=15000, image_size=None)

opi.records_db = opi.records_db[pd.isnull(opi.records_db['image_problems_from_text'])].reset_index(drop=True)
opi.records_db = opi.records_db[0:10000].reset_index(drop=True)

opi.pull(new_records_pull=False, download_limit=None)

# Check for grayscale
ip = ImageProcessing(opi)
ip.grayscale_analysis()


# Drop non-grayscale images
def delete_rule(row):
    if row['grayscale'] == False:
        return True

image_delete(ip, delete_rule=delete_rule)


# Note: From here, ~500 valid x-rays
# (i.e., images without arrows, grids, etc.) were selected by hand.
# The rest of the images were discarded.
