"""

    Harvest Data from the NIH's Open-i API
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    URL: https://openi.nlm.nih.gov.

"""
# Imports
import os
import re
import requests
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from tqdm import tqdm
from math import floor
from time import sleep
from scipy import misc
from pathlib import Path
from pprint import pprint
from itertools import chain
from datetime import datetime
from scipy.ndimage import imread as scipy_imread
from easymoney.easy_pandas import pandas_pretty_print # faze out

# Tool to create required caches
from biovida.init import _package_cache_creator

# Open-i API Parameters Information
from biovida.images.openi_parameters import openi_image_type_dict

# Tool for extracting features from text
from biovida.images.openi_text_feature_extraction import feature_extract

# BioVida Support Tools
from biovida.images.openi_support_tools import cln
from biovida.images.openi_support_tools import iter_join
from biovida.images.openi_support_tools import url_combine
from biovida.images.openi_support_tools import null_convert
from biovida.images.openi_support_tools import numb_extract
from biovida.images.openi_support_tools import item_extract
from biovida.images.openi_support_tools import extract_float
from biovida.images.openi_support_tools import filter_unnest
from biovida.images.openi_support_tools import num_word_to_int
from biovida.images.openi_support_tools import url_path_extract
from biovida.images.openi_support_tools import camel_to_snake_case
from biovida.images.openi_support_tools import openi_bounds_formatter

# To install scipy: brew install gcc; pip3 install Pillow

# Create Directories needed by the image package.
_package_cache_creator(sub_dir="image",
                       cache_path="/Users/tariq/Google Drive/Programming Projects/BioVida",
                       to_create=['raw', 'processed'])

# ---------------------------------------------------------------------------------------------
# General Terms
# ---------------------------------------------------------------------------------------------

search_query = 'retrieve.php?q=&it=c,m,mc,p,ph,u,x'

# ---------------------------------------------------------------------------------------------
# Pulling Data from the NIH's Open-i API
# ---------------------------------------------------------------------------------------------


def openi_bounds(total, req_limit=30):
    """

    :param total:
    :param req_limit:
    :return:
    """
    # Initalize
    end = 1
    bounds = list()

    # Block invalid values for 'total'.
    if total < 1:
        raise ValueError("'%s' is an invalid value for total" % (total))

    # Block the procedure below if total < req_limit
    if total < req_limit:
        return [(1, total)]

    # Compute the number of steps
    n_steps = int(floor(total/req_limit))

    # Floor the number of steps and loop
    for i in range(n_steps):
        bounds.append((end, end + (req_limit - 1)))
        end += req_limit

    # Compute the remainder
    remainder = total % req_limit

    # Add remaining part, if nonzero
    if remainder != 0:
        bounds += [(total - remainder + 1, total)]

    return bounds


def date_format(date_dict, date_format="%d/%m/%Y"):
    """

    :param date_dict:
    :param date_format:
    :return:
    """
    if not date_dict:
        return None

    info = [int(i) if i.isdigit() else None for i in [date_dict['year'], date_dict['month'], date_dict['day']]]
    if info[0] is None or (info[1] is None and info[2] is not None):
        return None

    cleaned_info = [1 if i is None or i < 1 else i for i in info]
    return datetime(cleaned_info[0], cleaned_info[1], cleaned_info[2]).strftime(date_format)


def harvest_vect(request_rslt):
    """

    Defines the terms to harvest from the results returned by the Open-i API.
    Assumes a maximum of one nest (i.e., request_rslt[key]: {key: value...}).

    :param request_rslt: Request Data from the server
    :return:
    """
    to_harvest = list()
    for k, v in request_rslt.items():
        if isinstance(v, str):
            to_harvest.append(k)
        elif isinstance(v, dict) and any(dmy in map(lambda x: x.lower(), v.keys()) for dmy in ['day', 'month', 'year']):
            to_harvest.append(k)
        elif isinstance(v, dict):
            for i in v.keys():
                to_harvest.append((k, i))

    return to_harvest


def openi_block_harvest(url, bound, root_url, to_harvest):
    """

    :param url:
    :param bound:
    :param root_url:
    :param to_harvest:
    :return:
    """
    # Init
    item_dict = dict()

    # Extract the starting point
    start = int(re.findall('&m=(.+?)&', bound)[0])

    # Request data from the Open-i servers
    req = requests.get(url + bound).json()['list']

    # Loop
    list_of_dicts = list()
    for e, item in enumerate(req):

        # Create an item_dict the dict
        item_dict = {"req_no": start + e}

        # Populate current `item_dict`
        for j in to_harvest:
            if isinstance(j, (list, tuple)):
                item_dict[iter_join(j)] = null_convert(item.get(j[0], {}).get(j[1], None))
            elif j == 'journal_date':
                item_dict[j] = null_convert(date_format(item.get(j, None)))
            elif 'img_' in camel_to_snake_case(j):
                item_dict[j] = url_combine(root_url, null_convert(item.get(j, None)))
            else:
                item_dict[j] = null_convert(item.get(j, None))

        list_of_dicts.append(item_dict)

    return list_of_dicts


def openi_harvest(bounds_list, joined_url, root_url, to_harvest, sleep_mini=(2, 5), sleep_main=(50, 300), verbose=True):
    """

    :param bounds_list:
    :param joined_url:
    :param bound:
    :param root_url:
    :param sleep_mini: (interval, sleep time in seconds)
    :param sleep_main: (interval, sleep time in seconds)
    :return:
    """
    harvested_data = list()
    print("\n---------------\nDownloading...\n---------------\n")
    for e, bound in enumerate(bounds_list, start=1):
        if verbose: # try printing every x downloads.
            print("Block %s of %s." % (e, len(bounds_list)))
        if e % sleep_mini[0] == 0:
            sleep(abs(sleep_mini[1] + np.random.normal()))
        elif e % sleep_main[0] == 0:
            if verbose:
                print("Sleeping for %s seconds..." % sleep_main[1])
            sleep(abs(sleep_main[1] + np.random.normal()))

        # Harvest
        harvested_data += openi_block_harvest(joined_url, bound, root_url, to_harvest)

    # Return
    return harvested_data


def openi_kinesin(search_query, req_limit=30, n_bounds_limit=5, verbose=True):
    """

    'Walk' along the search query and harvest the data.

    :param search_query:
    :param req_limit: Defaults to 30 (limit imposed by Open-i).
    :param n_bounds_limit: temp.
    :param verbose: print information.
    :return:
    """
    root_url = 'https://openi.nlm.nih.gov/'
    joined_url = root_url + search_query

    # Get a sample request
    sample = requests.get(joined_url + "&m=1&n=1").json()

    # Get the total number of results
    total = sample['total']

    # Block progress if no results found
    if total < 1:
        raise ValueError("No Results Found. Please Try Refining your Search Query.")

    # Print number of results found
    if verbose:
        print("\nResults Found: %s." % ('{:,.0f}'.format(total)))

    # Compute a list of search ranges to harvest
    bounds_list = openi_bounds_formatter(openi_bounds(total, req_limit=req_limit))

    # Define Extract Limit
    trunc_bounds = bounds_list[0:n_bounds_limit] if isinstance(n_bounds_limit, int) else bounds_list

    # Learn the data returned by the API
    to_harvest = harvest_vect(sample['list'][0])

    # Harvest and Convert to a DataFrame
    return pd.DataFrame(openi_harvest(trunc_bounds, joined_url, root_url, to_harvest)).fillna(np.NaN)



# ---------------------------------------------------------------------------------------------
# Construct Database
# ---------------------------------------------------------------------------------------------

def post_processing(data_frame):
    """

    :param data_frame:
    :return:
    """
    # Change camel case to snake case and lower
    data_frame.columns = list(map(lambda x: camel_to_snake_case(x).replace("me_sh", "mesh"), data_frame.columns))

    # Run Feature Extracting Tool and Join with `data_frame`.
    pp = pd.DataFrame(data_frame.apply(feature_extract, axis=1).tolist()).fillna(np.NaN)
    data_frame = df.join(pp, how='left')

    # Make the type of Imaging technology type human-readable
    data_frame['imaging_tech'] = data_frame['image_modality_major'].map(lambda x: openi_image_type_dict.get(x, np.NaN))
    del data_frame['image_modality_major']

    return data_frame

# Pull Data
df = openi_kinesin(search_query, n_bounds_limit=5)

# Run Post Processing
df = post_processing(df)

# ---------------------------------------------------------------------------------------------
# Image Harvesting
# ---------------------------------------------------------------------------------------------

# Start tqdm
tqdm.pandas(desc="status")

# thumb_reg; thumb_large; grid150; large


def img_name_abbrev(data_frame):
    """

    Returns a mapping of img types supplied by the Open-i API to abreviations

    Typical Results:
         - img_large: L
         - img_thumb: T
         - img_grid150: G150
         - img_thumb_large: TL

    :param data_frame: a dataframe returned by ``post_processing()``
    :return:
    """
    # Get the columns that contain df
    img_types = [i for i in data_frame.columns if "img_" in i]

    # Define a lambda to extract the first letter of all non 'img' 'words'.
    fmt = lambda x: "".join([j[0] + numb_extract(j) for j in x.split("_") if j != 'img']).upper()

    # Return a hash mapping
    return {k: fmt(k) for k in img_types}


img_abbreviations = img_name_abbrev(df)


def _img_titler(number, img_name, img_type, x_res, y_res, assumed_format='png'):
    """

    :param numbr:
    :param img_type:
    :param img_name:
    :param x_res:
    :param y_res:
    :return:
    """
    # Get Name
    img_name_title = img_name.split(".")[0]

    # Get image format (assume png if none)
    img_name_format = img_name.split(".")[1] if len(img_name.split(".")) == 2 else assumed_format

    # Generate the name
    new_name = "_".join(map(str, [number, img_name_title, img_type, x_res, y_res]))

    # Return
    return "{0}.{1}".format(new_name, img_name_format)



def img_harvest(image_address, save_location, lag=5.5):
    """Harvest Pics from a URL and save to disk"""
    # Save location
    save_address = "".join(map(str, [save_location, "__", image_address.split("/")[-1]]))

    # Check if the file already exists
    # If not, download and save it.
    page = None
    if not Path(save_address).is_file():
        try:
            page = requests.get(image_address)
        except:
            return False

        # Sleep
        sleep(abs(lag + np.random.normal()))

        # Save
        with open(save_address, 'wb') as img: # add resolution.
            img.write(page.content)

    return True



def bulk_img_harvest(data_frame, col1, db_map, save_location='images/'):
    """

    Bulk download of a set of images from the database

    :param data_frame:
    :param col1:
    :param db_map: column that can be used to map images back to a row in the database.
    :return:
    """
    # Download the images
    result_log = list()
    for x in tqdm(data_frame[col1]):
        result_log.append(img_harvest(x, save_location))


    # result_log = data_frame[col1].progress_map(lambda x: (x[1], img_harvest(x[0], db_mapping=x[1])), na_action='ignore')


# x = url_path_extract(data_frame[col1][0])

# ---------------------------------------------------------------------------------------------
# Pull Pictures and Represent them as numpy arrays.
# ---------------------------------------------------------------------------------------------


# Run the Harvesting tool
# bulk_img_harvest(data_frame=df, col1='img_large', db_map='req_no')



# Scheme: No_ImgType_ImageName_Resolution
# thumb_reg; thumb_large; grid150; large





# _img_titler("1",  "MPX1000.png", "T", "250", "302")




































df.img_grid150[pd.isnull(df.img_grid150)]




df.img_large[35]
df.img_grid150[35]


















