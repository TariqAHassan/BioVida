"""

    Harvest Data from the NIH's Open-i API
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    URL: https://openi.nlm.nih.gov.

    Need to extract Image Type (i.e., mark as: X-Ray, CT, PET, MRI, etc.)

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

# Open-i API Parameters Information
from biovida.images.openi_parameters import openi_image_type_dict

# Tool for extracting features from text
from biovida.images.text_feature_extraction import mexpix_info_extract
from biovida.images.text_feature_extraction import patient_sex_guess
from biovida.images.text_feature_extraction import age_refine
from biovida.images.text_feature_extraction import patient_age_guess
from biovida.images.text_feature_extraction import feature_extract

# BioVida Support Tools
from biovida.images.openi_support_tools import cln
from biovida.images.openi_support_tools import iter_join
from biovida.images.openi_support_tools import null_convert
from biovida.images.openi_support_tools import url_combine
from biovida.images.openi_support_tools import item_extract
from biovida.images.openi_support_tools import extract_float
from biovida.images.openi_support_tools import filter_unnest
from biovida.images.openi_support_tools import num_word_to_int
from biovida.images.openi_support_tools import openi_bounds_formatter

# To install scipy: brew install gcc; pip3 install Pillow

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
        bounds.append((end, end + 29))
        end += 30

    # Compute remainder
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


def openi_block_harvest(url, bound, root_url):
    """

    To Harvest:
      - title
      - affiliate
      - articleType
      - authors
      - uid
      - pmcid
      - journal_date (join these).
      - journal_title
      - licenseType
      - MeSH
          -- major
          -- minor
      - Problems
      - abstract
      - link: 'https://openi.nlm.nih.gov/' + imgLarge
      - fulltext_html_url
      - image
          -- captions
          -- id
          -- mention
          -- modalityMajor

    :param url:
    :param bound:
    :param root_url:
    :return:
    """
    # Init
    item_dict = dict()

    # Extract the starting point
    start = int(re.findall('&m=(.+?)&', bound)[0])

    # Define Items to harvest
    to_harvest = ['title', 'affiliate', 'articleType', 'authors', 'uid', 'pmcid',
                  'journal_date', 'journal_title', 'licenseType',
                  ('MeSH', 'major'), ('MeSH', 'minor'), 'Problems',
                  'abstract', 'imgLarge', 'fulltext_html_url', ('image', 'captions'),
                  ('image', 'id'), ('image', 'mention'), ('image', 'modalityMajor')]

    # Request Data from the server
    req = requests.get(url + bound).json()['list']

    # Loop
    list_of_dicts = list()
    for e, item in enumerate(req):
        # Init
        item_dict = {"req_no": start + e}

        # Populate current `item_dict`
        for j in to_harvest:
            if isinstance(j, tuple):
                item_dict[iter_join(j)] = null_convert(item.get(j[0], {}).get(j[1], None))
            elif j == 'journal_date':
                item_dict[j] = null_convert(date_format(item.get(j, None)))
            elif j == 'imgLarge':
                item_dict[j] = url_combine(root_url, null_convert(item.get(j, None)))
            else:
                item_dict[j] = null_convert(item.get(j, None))

        list_of_dicts.append(item_dict)

    return list_of_dicts


def openi_harvest(bounds_list, joined_url, root_url, sleep_mini=(2, 5), sleep_main=(50, 300), verbose=True):
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
    print("---------------\nDownloading...\n---------------\n")
    for e, bound in enumerate(bounds_list, start=1):
        if verbose:
            print("Block %s of %s." % (e, len(bounds_list)))
        if e % sleep_mini[0] == 0:
            sleep(abs(sleep_mini[1] + np.random.normal()))
        elif e % sleep_main[0] == 0:
            if verbose:
                print("Sleeping for %s seconds..." % sleep_main[1])
            sleep(abs(sleep_main[1] + np.random.normal()))

        # Harvest
        harvested_data += openi_block_harvest(joined_url, bound, root_url)

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

    # Get the total number of results
    total = requests.get(joined_url + "&m=1&n=1").json()['total']

    # Block progress if no results found
    if total < 1:
        raise ValueError("No Results Found.")

    # Print number of results found
    if verbose:
        print("Results Found: %s." % ('{:,.0f}'.format(total)))

    # Compute a list of search ranges to harvest
    bounds_list = openi_bounds_formatter(openi_bounds(total, req_limit=req_limit))

    # Define Extract Limit
    trunc_bounds = bounds_list[0:n_bounds_limit] if isinstance(n_bounds_limit, int) else bounds_list

    # Harvest
    harvested_data = openi_harvest(trunc_bounds, joined_url, root_url)

    # Convert to a DataFrame
    df = pd.DataFrame(harvested_data).fillna(np.NaN)

    # Lower the columns
    df.columns = list(map(lambda x: x.lower(), df.columns))

    # Return
    return df


# ---------------------------------------------------------------------------------------------
# Construct Database
# ---------------------------------------------------------------------------------------------

def post_processing(data_frame):
    """

    :param data_frame:
    :return:
    """
    # Run Feature Extracting Tool and Join with `data_frame`.
    pp = pd.DataFrame(data_frame.apply(feature_extract, axis=1).tolist()).fillna(np.NaN)
    data_frame = df.join(pp, how='left')

    # Make the type of Imaging technology type human-readable
    data_frame['imaging_tech'] = data_frame['image_modalitymajor'].map(lambda x: openi_image_type_dict.get(x, np.NaN))
    del data_frame['image_modalitymajor']

    return data_frame

# Pull Data
df = openi_kinesin(search_query, n_bounds_limit=5)

# Run Post Processing
df = post_processing(df)

# Start tqdm
tqdm.pandas(desc="status")

# ---------------------------------------------------------------------------------------------
# Image Harvesting
# ---------------------------------------------------------------------------------------------


def img_harvest(image_address, db_mapping, save_location='images/', lag=5.5):
    """Harvest Pics from a URL and save to disk"""
    # Save location
    save_address = "".join(map(str, [save_location, db_mapping, "__", image_address.split("/")[-1]]))

    # Check if the file already exists
    # If not, download and save it.
    page = None
    if not Path(save_address).is_file():
        try:
            # Get the image
            page = requests.get(image_address)
        except:
            return False

        # Sleep
        sleep(abs(lag + np.random.normal()))

        # Save
        with open(save_address, 'wb') as img:
            img.write(page.content)

    return True


def bulk_img_harvest(data_frame, col1, db_map, log_save_loc="images/logs/", save_interval=25):
    """

    Bulk download of a set of images from the database

    :param data_frame:
    :param col1:
    :param db_map: column that can be used to map images back to a row in the database.
    :param log_save_loc:
    :param save_interval:
    :return:
    """
    # Download the images
    save_name = lambda: datetime.now().strftime("%d-%m-%Y_%H:%M:%S_%s")
    result_log = list()
    for x in tqdm(zip(data_frame[col1], data_frame[db_map])):
        result_log.append((x[1], img_harvest(x[0], db_mapping=x[1])))
        if x[1] % save_interval == 0:
            pd.DataFrame(result_log, columns=[db_map, 'sucessful']).to_csv(log_save_loc + save_name() + "img_log.csv")


# ---------------------------------------------------------------------------------------------
# Pull Pictures and Represent them as numpy arrays.
# ---------------------------------------------------------------------------------------------


# Run the Harvesting tool
# bulk_img_harvest(df, col1='imglarge', db_map='req_no')




















































