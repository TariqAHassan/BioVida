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

from tqdm import tqdm
from math import floor
from time import sleep
from scipy import misc
from pathlib import Path
from copy import deepcopy
from pprint import pprint
from itertools import chain
from datetime import datetime
from scipy.ndimage import imread as scipy_imread
from easymoney.easy_pandas import pandas_pretty_print # faze out

# Tool to create required caches
from biovida.init import _package_cache_creator

# Open-i API Parameters Information
from biovida.images.openi_parameters import openi_image_type_params
from biovida.images.openi_parameters import openi_search_information

# Tool for extracting features from text
from biovida.images.openi_text_feature_extraction import feature_extract

# BioVida Support Tools
from biovida.images.openi_support_tools import cln
from biovida.images.openi_support_tools import header
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
from biovida.images.openi_support_tools import list_to_bulletpoints
from biovida.images.openi_support_tools import openi_bounds_formatter

# To install scipy: brew install gcc; pip3 install Pillow

# Create Directories needed by the image package.
root_img_path, created_img_dirs = _package_cache_creator(sub_dir="image",
                                                         cache_path="/Users/tariq/Google Drive/Programming Projects/BioVida",
                                                         to_create=['raw', 'processed'])


# ToDo:
#   - Add option to cache a search.



def _openi_search_special_case(search_param, blocked, passed):
    """

    :param search_param: one of 'video', 'image_type'...
    :param blocked: muturally exclusive (i.e., all these items cannot be passed together).
    :param passed: values actually passed to `search_param`.
    :return:
    """
    if all(b in passed for b in blocked):
        raise ValueError("`%s` can only contain one of:\n%s" % (search_param, list_to_bulletpoints(blocked)))


def _openi_search_check(search_arguments, search_dict):
    """

    :param args:
    :return: ``None``
    """
    general_error_msg = "'{0}' is not valid for `{1}`.\n`{1}` must be one of:\n{2}"

    # Check `query`
    if not isinstance(search_arguments['query'], str):
        raise ValueError("`query` must be a string.")

    # Check all other params
    for k, v in search_arguments.items():
        # Hault if query or `v` is NoneType
        if k != 'query' and v is not None:

            # Check type
            if not isinstance(v, (list, tuple)) and v is not None:
                raise ValueError("Only `lists`, `tuples` or `None` may be passed to `%s`." % (k))

            # Loop though items in `v`
            for i in v:
                if not isinstance(i, str):
                    raise ValueError("`tuples` or `lists` passed to `%s` must only contain strings." % (k))
                # Check if `i` can be converted to a param understood by the Open-i API
                if i not in search_dict[k][1].keys():
                    raise ValueError(general_error_msg.format(i, k, list_to_bulletpoints(search_dict[k][1].keys())))

            # Block contradictory requests
            if k == 'video':
                _openi_search_special_case(k, blocked=['true', 'false'], passed=v)
            if k == 'rankby':
                _openi_search_special_case(k, blocked=['newest', 'oldest'], passed=v)


def _url_formatter(api_search_transform, ordered_params):
    """

    :param ordered_params:
    :param api_search_transform:
    :return:
    """
    search_term = ""
    for p in ordered_params:
        if p in api_search_transform:
            search_term += "{0}{1}={2}".format(("?" if p == 'q' else ""), p, api_search_transform[p])

    return "https://openi.nlm.nih.gov/retrieve.php{0}".format(search_term)


def _exclusions_img_type_merge(args, exclusions):
    """Merge Image type with Exclusions"""
    # Check `exclusions` is an acceptable type
    if not isinstance(exclusions, (list, tuple)) and exclusions is not None:
        raise ValueError('`exclusions` must be a `list`, `tuple` or `None`.')

    # Return if there is nothing to check
    if exclusions is None or (isinstance(exclusions, (list, tuple)) and not len(exclusions)):
        return args

    # Check exclusions only contains allowed types
    if any(e not in ['graphics', 'multipanel'] for e in exclusions):
        raise ValueError("`exclusions` must only include one or all of: 'graphics', 'multipanel'.")

    # Handle handle tuples, then `None`s.
    args['image_type'] = list(args['image_type']) if isinstance(args['image_type'], (list, tuple)) else []

    # Merge `exclusions` with `imgage_type`
    args['image_type'] += list(map(lambda x: 'exclude_{0}'.format(x), exclusions))

    return args


def openi_search_options(option, print_options=True):
    """

    Options for parameters of `openi_search()`.

    :param option: one of: 'image_type', 'rankby', 'subset', 'collection', 'fields', 'specialties', 'video'.
    :param print: if True, pretty print the options, else return.
    :return:
    """
    # Terms to blocked from displaying
    blocked = ['exclude_graphics', 'exclude_multipanel']

    # Get the relevant dict of params
    search_dict = openi_search_information()[0].get(cln(option).strip().lower(), None)

    # Report invalid `option`
    if search_dict is None:
        raise ValueError("'{0}' is not a valid option for the Open-i API".format(option))

    # Remove blocked term
    opts = [i for i in search_dict[1].keys() if i not in blocked]
    if not len(opts):
        raise ValueError("Relevant options for '{0}'.".format(option))

    # Print or Return
    if print_options:
        print(list_to_bulletpoints(opts))
    else:
        return list(opts)


def openi_search(query
                 , image_type=None
                 , rankby=None
                 , subset=None
                 , collection=None
                 , fields=None
                 , specialties=None
                 , video=['false']
                 , exclusions=['graphics']):
    """

    Tool to generate search terms for the NIH's Open-i API.

    :param query: a search term. ``None`` will converter to an empty string.
    :type query: ``str`` or ``None``
    :param image_type: see `search_options('image_type')` for valid values.
    :type image_type: ``list``, ``tuple`` or ``None``.
    :param rankby: see `search_options('rankby')` for valid values.
    :type rankby: ``list``, ``tuple`` or ``None``.
    :param subset: see `search_options('subset')` for valid values.
    :type subset: ``list``, ``tuple`` or ``None``.
    :param collection: see `search_options('collection')` for valid values.
    :type collection: ``list``, ``tuple`` or ``None``.
    :param fields: see `search_options('fields')` for valid values.
    :type fields: ``list``, ``tuple`` or ``None``.
    :param specialties: see `search_options('specialties')` for valid values.
    :type specialties: ``list``, ``tuple`` or ``None``.
    :param video: see `search_options('video')` for valid values. Defaults to ['false'].
    :type video: ``list``, ``tuple`` or ``None``.
    :param exclusions: one or both of: 'graphics', 'multipanel'. Defaults to ['graphics'].
    :type exclusions: ``list``, ``tuple`` or ``None``
    :return: search url for the Open-i API.
    :rtype: ``str``
    """
    # Block blank searchers
    if not len(list(filter(None, locals().values()))):
        raise ValueError("No Search Criterion Detected. Please specify criterion to narrow your search.")

    # Extract the function arguments
    args = {k: v for k, v in deepcopy(locals()).items() if k != 'exclusions'}

    # Merge Image type with Exclusions
    args = _exclusions_img_type_merge(args, exclusions)

    # Define a lambda to clean the search terms
    search_clean = lambda k, v: [cln(i).strip().replace(' ', '_').lower() for i in v] if k != 'query' and v is not None else v

    # Get the arguments
    search_arguments = {k: '' if k == 'query' and v is None else search_clean(k, v) for k, v in args.items()}

    # Get Open-i Search Params
    search_dict, ordered_params = openi_search_information()

    # Add check for all search terms
    _openi_search_check(search_arguments, search_dict)

    # Convert argument names into a form the API will understand
    api_url_param = lambda x: search_dict[x][0] if x != 'query' else 'q'

    # Convert values passed into a form the API will understand
    api_url_terms = lambda k, v: ','.join([search_dict[k][1][i] for i in v]) if k != 'query' else v

    # Perform transformation
    api_search_transform = {api_url_param(k): api_url_terms(k, v) for k, v in search_arguments.items() if v is not None}

    # Format `api_search_transform` and return
    return _url_formatter(api_search_transform, ordered_params)



# ---------------------------------------------------------------------------------------------
# General Terms
# ---------------------------------------------------------------------------------------------

search_query = 'https://openi.nlm.nih.gov/retrieve.php?q=&it=c,m,mc,p,ph,u,x'

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
    header("Downloading Records... ")
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
    # Get a sample request
    sample = requests.get(search_query + "&m=1&n=1").json()

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
    return pd.DataFrame(openi_harvest(trunc_bounds, search_query, root_url, to_harvest)).fillna(np.NaN)


# ---------------------------------------------------------------------------------------------
# Image Harvesting
# ---------------------------------------------------------------------------------------------

# Start tqdm
tqdm.pandas(desc="status")


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


def _img_titler(number, img_name, img_type, assumed_format='png'):
    """

    :param number:
    :param img_type:
    :param img_name:
    :return:
    """
    # Get Name
    img_name_title = img_name.split(".")[0]

    # Get image format (assume png if none)
    img_name_format = img_name.split(".")[1] if len(img_name.split(".")) == 2 else assumed_format

    # Generate the name
    new_name = "__".join(map(str, [number, img_name_title, img_type]))

    # Return
    return "{0}.{1}".format(new_name, img_name_format)


def img_harvest(img_title, image_web_address, image_save_location, lag=5.5):
    """

    Harvest Pics from a URL and save to disk.

    :param image_web_address:
    :param image_save_location:
    :param lag:
    :return:
    """
    # Init
    page = None

    # Define the save path
    image_save_path = os.path.join(image_save_location, img_title)

    # Check if the file already exists; if not, download and save it.
    if not Path(image_save_path).is_file():
        try:
            # Get the image
            page = requests.get(image_web_address)

            # Sleep
            sleep(abs(lag + np.random.normal()))

            # Save to disk
            with open(image_save_path, 'wb') as img:
                img.write(page.content)
        except:
            return False

    return True


def bulk_img_harvest(data_frame, image_column, image_save_location=created_img_dirs['raw'], verbose=False):
    """

    Bulk download of a set of images from the database.

    :param data_frame:
    :param image_column:
    :param image_save_location:
    :param verbose:
    :return:
    """
    # Log of Sucessess
    result_log = dict()

    # Get the abbreviation for the image type being downloaded
    img_type = img_name_abbrev(data_frame)[image_column]

    # Download the images
    header("Downloading Images... ")
    for img_address in tqdm(data_frame[image_column]):
        # Name the image
        img_title = _img_titler(number=1, img_name=img_address.split("/")[-1], img_type=img_type)

        # Try download and log whether or not Download was sucessful.
        result_log[img_address] = img_harvest(img_title, img_address, image_save_location)

    if verbose:
        failed_downloads = {k: v for k, v in result_log.items() if v is False}
        if len(failed_downloads.values()):
            header("Failed Downloads: ")
            for k in failed_downloads.keys():
                print(" - " + k)
        else:
            print("All Images Sucessfully Extracted.")

    # Map record of download sucess to img_address (URLs).
    sucesses_log = data_frame[image_column].map(lambda x: result_log.get(x, np.NaN) if pd.notnull(x) else np.NaN)

    # Return sucesses log
    return sucesses_log.rename("extracted")


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
    data_frame = data_frame.join(pp, how='left')

    # Make the type of Imaging technology type human-readable
    data_frame['imaging_tech'] = data_frame['image_modality_major'].map(lambda x: openi_image_type_params.get(x, np.NaN))
    del data_frame['image_modality_major']

    return data_frame


def openi_pull(search_query, image_quality='large', n_bounds_limit=1, verbose=False):
    """


    :param search_query:
    :param image_quality: one of: large, grid150, thumb, thumb_large or ``None``. Defaults to Large.
                          If None, Images will not be harvested.
    :param n_bounds_limit:
    :param verbose:
    :return:
    """
    allowed_image_quality_types = ['large', 'grid150', 'thumb', 'thumb_large']
    if image_quality is not None and image_quality not in allowed_image_quality_types:
        raise ValueError("`image_quality` must be one of: %s" % \
                         (", ".join(map(lambda x: "'{0}'".format(x), allowed_image_quality_types))))

    # Pull Data
    data_frame = openi_kinesin(search_query, n_bounds_limit=n_bounds_limit, verbose=verbose)

    # Run Post Processing
    data_frame = post_processing(data_frame)

    # Download Images
    if image_quality is not None:
        data_frame['img_extracted'] = bulk_img_harvest(data_frame, "img_{0}".format(image_quality))

    return data_frame


class OpenInterface(object):
    """

    """
    def __init__(self):
        """

        """

    def search(self):
        """

        :return:
        """

    def pull(self):
        """

        :return:
        """



























