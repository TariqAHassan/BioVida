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
from warnings import warn
from copy import deepcopy
from pprint import pprint
from itertools import chain
from datetime import datetime
# from scipy.ndimage import imread as scipy_imread
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

# To install scipy: brew install gcc; pip3 install Pillow

# Start tqdm
tqdm.pandas(desc='status')


# ToDo: Add the ability to cache a search.

# test search
search_query = 'https://openi.nlm.nih.gov/retrieve.php?q=&it=c,m,mc,p,ph,u,x'

# ---------------------------------------------------------------------------------------------
# Open-i Searching
# ---------------------------------------------------------------------------------------------

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
    # Format query
    if 'query' in api_search_transform:
        api_search_transform['query'] = cln(api_search_transform['query']).replace(' ', '+')
    else:
        raise ValueError("No `query` detected.")

    search_term = ""
    for p in ordered_params:
        if p in api_search_transform:
            search_term += "{0}{1}={2}".format(("?" if p == 'query' else ""), p, api_search_transform[p])

    # ToDo: change to root_url
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

    # Handle handle tuples, then `None`s (with the 'else []').
    args['image_type'] = list(args['image_type']) if isinstance(args['image_type'], (list, tuple)) else []

    # Merge `exclusions` with `imgage_type`
    args['image_type'] += list(map(lambda x: 'exclude_{0}'.format(x), exclusions))

    return args

# ---------------------------------------------------------------------------------------------
# Pull Records from the NIH's Open-i API
# ---------------------------------------------------------------------------------------------

class _OpeniRecords(object):
    """

    Tools to Pull Records from the Open-i API.

    """


    def __init__(self, root_url, date_format, n_bounds_limit, sleep_mini, sleep_main, verbose, req_limit=30):
        """

        :param root_url:                                            Suggested: 'https://openi.nlm.nih.gov'
        :param date_format: Defaults to '%d/%m/%Y'                  Suggested: "%d/%m/%Y" (consider leaving as datatime)
        :param n_bounds_limit:                                      Suggested: 5
        :param sleep_mini: (interval, sleep time in seconds)        Suggested: (2, 5)
        :param sleep_main: (interval, sleep time in seconds)        Suggested: (50, 300)
        :param verbose: print additional detail.                    Suggested: True
        :param req_limit: Defaults to 30.                           Required by Open-i: 30
        """
        self.root_url = root_url
        self.date_format = date_format
        self.n_bounds_limit = n_bounds_limit
        self.sleep_mini = sleep_mini
        self.sleep_main = sleep_main
        self.verbose = verbose
        self.req_limit = req_limit


    def openi_bounds(self, total):
        """

        :param total:
        :return:
        """
        # Initalize
        end = 1
        bounds = list()

        # Block invalid values for 'total'.
        if total < 1:
            raise ValueError("'%s' is an invalid value for total" % (total))

        # Block the procedure below if total < req_limit
        if total < self.req_limit:
            return [(1, total)]

        # Compute the number of steps
        n_steps = int(floor(total/self.req_limit))

        # Floor the number of steps and loop
        for i in range(n_steps):
            bounds.append((end, end + (self.req_limit - 1)))
            end += self.req_limit

        # Compute the remainder
        remainder = total % self.req_limit

        # Add remaining part, if nonzero
        if remainder != 0:
            bounds += [(total - remainder + 1, total)]

        return bounds


    def openi_bounds_formatter(self, bounds):
        """

        Format the computed bounds for the Open-i API.

        :param bounds: as returned by `_OpeniPull().openi_bounds()`
        :return:
        :rtype: ``list``
        """
        return ["&m={0}&n={1}".format(i[0], i[1]) for i in bounds]


    def date_formater(self, date_dict):
        """

        :param date_dict:
        :param date_format:
        :return:
        """
        if not date_dict:
            return None

        try:
            info = [int(i) if i.isdigit() else None for i in [date_dict['year'], date_dict['month'], date_dict['day']]]
        except:
            return None
        if info[0] is None or (info[1] is None and info[2] is not None):
            return None

        cleaned_info = [1 if i is None or i < 1 else i for i in info]
        return datetime(cleaned_info[0], cleaned_info[1], cleaned_info[2]).strftime(self.date_format)


    def harvest_vect(self, request_rslt):
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


    def openi_block_harvest(self, url, bound, to_harvest):
        """

        :param url:
        :param bound:
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
                    item_dict[j] = null_convert(self.date_formater(item.get(j, None)))
                elif 'img_' in camel_to_snake_case(j):
                    item_dict[j] = url_combine(self.root_url, null_convert(item.get(j, None)))
                else:
                    item_dict[j] = null_convert(item.get(j, None))

            list_of_dicts.append(item_dict)

        return list_of_dicts


    def openi_harvest(self, bounds_list, joined_url, to_harvest):
        """

        :param bounds_list:
        :param joined_url:
        :param bound:
        :return:
        """
        # Init
        c = 1
        harvested_data = list()

        # Print Header
        header("Downloading Records... ")

        # Print updates
        if self.verbose:
            print("\nNumber of Records to Download: {0} (block size: {1} rows max).".format(
                str(int(self.req_limit * len(bounds_list))), str(self.req_limit))
            )

        for bound in tqdm(bounds_list):
            c += 1
            if c % self.sleep_mini[0] == 0:
                sleep(abs(self.sleep_mini[1] + np.random.normal()))
            elif c % self.sleep_main[0] == 0:
                if self.verbose:
                    print("Sleeping for %s seconds..." % self.sleep_main[1])
                sleep(abs(self.sleep_main[1] + np.random.normal()))

            # Harvest
            harvested_data += self.openi_block_harvest(joined_url, bound, to_harvest)

        # Return
        return harvested_data


    def openi_kinesin(self, search_query, to_harvest, total):
        """

        'Walk' along the search query and harvest the data.

        :param search_query:
        :param total:
        :return:
        """
        # Compute a list of search ranges to harvest
        bounds_list = self.openi_bounds_formatter(self.openi_bounds(total))

        # Define Extract Limit
        trunc_bounds = bounds_list[0:self.n_bounds_limit] if isinstance(self.n_bounds_limit, int) else bounds_list

        # Learn the data returned by the API
        to_harvest = self.harvest_vect(to_harvest)

        # Harvest and Convert to a DataFrame
        return pd.DataFrame(self.openi_harvest(trunc_bounds, search_query, to_harvest)).fillna(np.NaN)


# ---------------------------------------------------------------------------------------------
# Image Harvesting
# ---------------------------------------------------------------------------------------------

class _OpeniImages(object):
    """

    """



    def __init__(self, assumed_img_format, sleep_time, image_save_location, verbose):
        """


        :param assumed_img_format: suggested: 'png'
        :param sleep_time: suggested: 5.5
        :param image_save_location: suggested: created_img_dirs['raw'])
        :param verbose:
        """
        self.assumed_img_format = assumed_img_format
        self.sleep_time = sleep_time
        self.image_save_location = image_save_location
        self.verbose = verbose


    def img_name_abbrev(self, data_frame):
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


    def img_titler(self, number, img_name, img_type):
        """

        :param number:
        :param img_type:
        :param img_name:
        :return:
        """
        # Get Name
        img_name_title = img_name.split(".")[0]

        # Get image format (assume png if none)
        img_name_format = img_name.split(".")[1] if len(img_name.split(".")) == 2 else self.assumed_img_format

        # Generate the name
        new_name = "__".join(map(str, [number, img_name_title, img_type]))

        # Return
        return "{0}.{1}".format(new_name, img_name_format)


    def img_harvest(self, img_title, image_web_address):
        """

        Harvest Pics from a URL and save to disk.

        :param image_web_address:
        :param lag:
        :return:
        """
        # Init
        page = None

        # Define the save path
        image_save_path = os.path.join(self.image_save_location, img_title)

        # Check if the file already exists; if not, download and save it.
        if not Path(image_save_path).is_file():
            try:
                # Get the image
                page = requests.get(image_web_address)

                # Sleep
                sleep(abs(self.sleep_time + np.random.normal()))

                # Save to disk
                with open(image_save_path, 'wb') as img:
                    img.write(page.content)
            except:
                return False

        return True


    def bulk_img_harvest(self, data_frame, image_column):
        """

        Bulk download of a set of images from the database.

        :param data_frame:
        :param image_column:
        :return:
        """
        # Log of Sucessess
        result_log = dict()

        # Get the abbreviation for the image type being downloaded
        img_type = self.img_name_abbrev(data_frame)[image_column]

        # Download the images
        header("Downloading Images... ")
        for img_address in tqdm(data_frame[image_column]):
            # Name the image
            img_title = self.img_titler(number=1, img_name=img_address.split("/")[-1], img_type=img_type)

            # Try download and log whether or not Download was sucessful.
            result_log[img_address] = self.img_harvest(img_title, img_address)

        if self.verbose:
            failed_downloads = {k: v for k, v in result_log.items() if v is False}
            if len(failed_downloads.values()):
                header("Failed Downloads: ")
                for k in failed_downloads.keys():
                    print(" - " + k)
            else:
                print("\nAll Images Sucessfully Extracted.")

        # Map record of download sucess to img_address (URLs).
        sucesses_log = data_frame[image_column].map(lambda x: result_log.get(x, None) if pd.notnull(x) else None)

        # Return sucesses log
        return sucesses_log.rename("extracted")

# ---------------------------------------------------------------------------------------------
# Construct Database
# ---------------------------------------------------------------------------------------------

class OpenInterface(object):
    """

    """


    def __init__(self
                 , cache_path=None
                 , n_bounds_limit=2
                 , img_sleep_time=5.5
                 , image_quality='large'
                 , date_format="%d/%m/%Y"
                 , assumed_img_format='png'
                 , records_sleep_mini=(2, 5)
                 , records_sleep_main=(50, 300)
                 , verbose=True):
        """


        :param cache_path:
        :param n_bounds_limit: max. number of blocks to download (1 block = 30 records/rows).
                               If ``None``, no limit will be imposed (not recommended). Defaults to 2.
        :param img_sleep_time:
        :param image_quality: one of: 'large', 'grid150', 'thumb', 'thumb_large' or ``None``. Defaults to 'large'.
                              If ``None``, no attempt will be made to download images.
        :param date_format:
        :param assumed_img_format:
        :param records_sleep_mini:
        :param records_sleep_main:
        :param verbose:
        """
        self._n_bounds_limit = n_bounds_limit
        self._verbose = verbose
        root_url = 'https://openi.nlm.nih.gov'

        # Define allowed image types to pull
        allowed_image_quality_types = ['large', 'grid150', 'thumb', 'thumb_large']

        # Check `image_quality` is valid
        if image_quality is not None and image_quality not in allowed_image_quality_types:
            raise ValueError("`image_quality` must be one of: %s" % \
                             (", ".join(map(lambda x: "'{0}'".format(x), allowed_image_quality_types))))
        self._image_quality = image_quality

        # Generate Required Caches
        pcc = _package_cache_creator(sub_dir='image', cache_path=cache_path, to_create=['raw', 'processed'])
        self._root_img_path, self._created_img_dirs = pcc

        # Create an instance of the _OpeniRecords() Class
        self._OpeniRecords = _OpeniRecords(root_url=root_url
                                           , date_format=date_format
                                           , n_bounds_limit=n_bounds_limit
                                           , sleep_mini=records_sleep_mini
                                           , sleep_main=records_sleep_main
                                           , verbose=verbose)

        # Create an instance of the _OpeniImages() Class
        self._OpeniImages = _OpeniImages(assumed_img_format=assumed_img_format
                                         , sleep_time=img_sleep_time
                                         , image_save_location=self._created_img_dirs['raw']
                                         , verbose=verbose)

        # Define the current search term
        self.current_search = None
        self.current_search_url = None
        self.current_search_total = None
        self._current_search_to_harvest = None


    def _post_processing_text(self, data_frame):
        """

        :param data_frame:
        :return:
        """
        # ToDo: add 'article_type' to dict. look-up

        # Change camel case to snake case and lower
        data_frame.columns = list(map(lambda x: camel_to_snake_case(x).replace("me_sh", "mesh"), data_frame.columns))

        # Run Feature Extracting Tool and Join with `data_frame`.
        pp = pd.DataFrame(data_frame.apply(feature_extract, axis=1).tolist()).fillna(np.NaN)
        data_frame = data_frame.join(pp, how='left')

        # Make the type of Imaging technology type human-readable
        data_frame['imaging_tech'] = data_frame['image_modality_major'].map(
            lambda x: openi_image_type_params.get(x, np.NaN)
        )
        del data_frame['image_modality_major']

        return data_frame


    def _search_probe(self, search_query, print_results):
        """

        :param search_query:
        :type search_query: ``str``
        :param print_results:
        :type print_results: ``bool``
        :return:
        """
        # Get a sample request
        sample = requests.get(search_query + "&m=1&n=1").json()

        # Get the total number of results
        try:
            total = int(float(sample['total']))
        except:
            raise ValueError("Could not obtain total number of results from the Open-i API.")

        # Block progress if no results found
        if total < 1:
            raise ValueError("No Results Found. Please Try Refining your Search.")

        # Print number of results found
        if print_results:
            print("\nResults Found: %s." % ('{:,.0f}'.format(total)))

        return total, sample['list'][0]


    def search_options(self, option, print_options=True):
        """

        Options for parameters of `openi_search()`.

        :param option: one of: 'image_type', 'rankby', 'subset', 'collection', 'fields', 'specialties', 'video'.
        :param print: if True, pretty print the options, else return as a ``list``.
        :return:
        """
        # Terms to blocked from displaying to users
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
            return opts


    def search(self
               , query
               , image_type=None
               , rankby=None
               , subset=None
               , collection=None
               , fields=None
               , specialties=None
               , video=None
               , exclusions=['graphics']
               , print_results=True):
        """

        Tool to generate a search term for the NIH's Open-i API.

        :param query: a search term. ``None`` will converter to an empty string.
        :type query: ``str`` or ``None``
        :param image_type: see `OpenInterface().search_options('image_type')` for valid values.
        :type image_type: ``list``, ``tuple`` or ``None``.
        :param rankby: see `OpenInterface().search_options('rankby')` for valid values.
        :type rankby: ``list``, ``tuple`` or ``None``.
        :param subset: see `OpenInterface().search_options('subset')` for valid values.
        :type subset: ``list``, ``tuple`` or ``None``.
        :param collection: see `OpenInterface().search_options('collection')` for valid values.
        :type collection: ``list``, ``tuple`` or ``None``.
        :param fields: see `OpenInterface().search_options('fields')` for valid values.
        :type fields: ``list``, ``tuple`` or ``None``.
        :param specialties: see `OpenInterface().search_options('specialties')` for valid values.
        :type specialties: ``list``, ``tuple`` or ``None``.
        :param video: see `OpenInterface().search_options('video')` for valid values. Defaults to ``None``.
        :type video: ``list``, ``tuple`` or ``None``.
        :param exclusions: one or both of: 'graphics', 'multipanel'. Defaults to ['graphics'].      -- Working?
        :type exclusions: ``list``, ``tuple`` or ``None``
        :param print_results: if ``True``, print the number of search results.
        :type print_results: ``bool``
        :return: search URL for the Open-i API.
        :rtype: ``str``
        """
        # Remove 'self' from locals
        args_cleaned = {k: v for k, v in deepcopy(locals()).items() if k not in ['self', 'print_results']}

        # Block blank searches
        if not len(list(filter(None, args_cleaned.values()))):
            raise ValueError("No Search Criterion Detected. Please specify criterion to narrow your search.")

        # Extract the function arguments
        args = {k: v for k, v in args_cleaned.items() if k != 'exclusions'}

        # Merge `image_type` with `exclusions`
        args = _exclusions_img_type_merge(args, exclusions)

        # Define a lambda to clean the search terms
        search_clean = lambda k, v: [cln(i).replace(' ', '_').lower() for i in v] if k != 'query' and v is not None else v

        # Get the arguments
        search_arguments = {k: '' if k == 'query' and v is None else search_clean(k, v) for k, v in args.items()}

        # Save search query
        self.current_search = {k: v for k, v in deepcopy(search_arguments).items() if v is not None}

        # Get Open-i search params
        search_dict, ordered_params = openi_search_information()

        # Add check for all search terms
        _openi_search_check(search_arguments, search_dict)

        # Convert param names into a form the API will understand
        api_url_param = lambda x: search_dict[x][0] if x != 'query' else 'query'

        # Convert values passed into a form the API will understand
        api_url_terms = lambda k, v: ','.join([search_dict[k][1][i] for i in v]) if k != 'query' else v

        # Perform transformation
        api_search_transform = {api_url_param(k): api_url_terms(k, v) for k, v in search_arguments.items() if v is not None}

        # Format `api_search_transform`
        formatted_search = _url_formatter(api_search_transform, ordered_params)

        # Save `formatted_search`
        self.current_search_url = formatted_search
        self.current_search_total, self._current_search_to_harvest = self._search_probe(formatted_search, print_results)


    def pull(self):
        """

        Pull the current search.

        :return: a DataFrame with the record information.
                 If `image_quality` is not None, images will also be harvested and cached.
        :rtype: ``Pandas DataFrame``
        """
        if self.current_search_url is None:
            raise ValueError("A search has not been defined. Please call `OpenInterface().search()`.")

        # Pull Data
        data_frame = self._OpeniRecords.openi_kinesin(self.current_search_url
                                                      , to_harvest=self._current_search_to_harvest
                                                      , total=self.current_search_total)

        # Run Post Processing
        data_frame = self._post_processing_text(data_frame)

        # Download Images
        if self._image_quality is not None:
            image_col = "img_{0}".format(self._image_quality)
            data_frame['img_extracted'] = self._OpeniImages.bulk_img_harvest(data_frame, image_column=image_col)
        elif self._verbose:
            warn("\nNo attempt was made to download images because `image_quality` is `None`.")

        return data_frame


























