"""

    Convnet Training Images Directly from Open-i
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from collections import Counter

from biovida.images import image_divvy
from biovida.images import OpeniInterface
from biovida.images import ImageProcessing
from biovida.support_tools import pandas_pprint

from biovida.support_tools.support_tools import cln

from data.source_data._private.my_file_paths import target_dir

# Start tqdm
tqdm.pandas(desc='status')


# ----------------------------------------------------------------------------------------------------------
# Harvesting
# ----------------------------------------------------------------------------------------------------------


opi = OpeniInterface()
# opi.search(image_type=['mri', 'ct', 'ultrasound', 'x_ray'])
# df = opi.pull(download_limit=None, image_size=None, clinical_cases_only=False)

ip = ImageProcessing(opi, db_to_extract='cache_records_db')
ip.grayscale_analysis()


df = ip.image_dataframe_short[ip.image_dataframe_short['grayscale'] == True].copy()
Counter(df['image_problems_from_text'].tolist())


def my_divvy_rule(row):
    if row['grayscale'] == True:
        return row['image_problems_from_text'][0]


train_val_test_dict = {'train': 0.8, 'validation': 0.1, 'test': 0.1, 'target_dir': target_dir}
tvt = image_divvy(ip, divvy_rule=my_divvy_rule, action='copy', train_val_test_dict=train_val_test_dict)















































