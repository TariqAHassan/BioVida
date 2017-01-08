"""

    Open-i Interface
    ~~~~~~~~~~~~~~~~


"""
# Imports
import os
import re
import pickle
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from math import floor
from time import sleep
from warnings import warn
from copy import deepcopy
from pprint import pprint
from datetime import datetime
from collections import Counter

# Tool to create required caches
from biovida.support_tools._cache_management import _package_cache_creator

# Open-i API Parameters Information
from biovida.images.openi_parameters import openi_image_type_params
from biovida.images.openi_parameters import openi_search_information
from biovida.images.openi_parameters import openi_article_type_params

# Tool for extracting features from text
from biovida.images.text_processing import feature_extract

# Image Support Tools
from biovida.images.openi_support_tools import iter_join
from biovida.images.openi_support_tools import url_combine
from biovida.images.openi_support_tools import null_convert
from biovida.images.openi_support_tools import numb_extract

# General Support Tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import same_dict
from biovida.support_tools.support_tools import unique_dics
from biovida.support_tools.support_tools import hashable_cols
from biovida.support_tools.support_tools import camel_to_snake_case
from biovida.support_tools.support_tools import list_to_bulletpoints

# To install scipy: brew install gcc; pip3 install Pillow

# Start tqdm
tqdm.pandas(desc='status')


# ---------------------------------------------------------------------------------------------
# Pull Records from the NIH's Open-i API
# ---------------------------------------------------------------------------------------------


class _OpeniRecords(object):
    """

    Tools to Pull Records from the Open-i API.

    """

    def __init__(self, root_url, date_format, download_limit, sleep_mini, sleep_main, verbose, req_limit=30):
        """

        :param root_url:                                            Suggested: 'https://openi.nlm.nih.gov'
        :param date_format:                                         Suggested: "%d/%m/%Y" (consider leaving as datatime)
        :param download_limit:                                      Suggested: 60
        :param sleep_mini: (interval, sleep time in seconds)        Suggested: (2, 5)
        :param sleep_main: (interval, sleep time in seconds)        Suggested: (50, 300)
        :param verbose: print additional details.                   Suggested: True
        :param req_limit: Defaults to 30.                           Required by Open-i: 30
        """
        self.root_url = root_url
        self.date_format = date_format
        self.download_limit = download_limit
        self.sleep_mini = sleep_mini
        self.sleep_main = sleep_main
        self.verbose = verbose
        self.req_limit = req_limit

    def openi_bounds(self, total):
        """

        :param total: the total number of results for a given search.
        :type total: int
        :return:
        """
        # Initalize
        end = 1
        bounds = list()

        # Block invalid values for 'total'.
        if total < 1:
            raise ValueError("'{0}' is an invalid value for total.".format(str(total)))

        # Check `self.download_limit`
        if self.download_limit is not None and not isinstance(self.download_limit, int):
            raise ValueError("`download_limit` must be an `int` or `None`.")
        if isinstance(self.download_limit, int) and self.download_limit < 1:
            raise ValueError("`download_limit` cannot be less than 1.")

        # Check total
        if total < self.req_limit:
            return [(1, total)]
        elif self.download_limit is not None and total > self.download_limit:
            download_no = self.download_limit
        else:
            download_no = total

        # Compute the number of steps and floor
        n_steps = int(floor(download_no / self.req_limit))

        # Loop through the steps
        for i in range(n_steps):
            bounds.append((end, end + (self.req_limit - 1)))
            end += self.req_limit

        # Compute the remainder
        remainder = download_no % self.req_limit

        # Add remaining part, if nonzero
        if remainder != 0:
            bounds += [(download_no - remainder + 1, download_no)]

        return bounds, download_no

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

        # ToDo: add date format guessing.
        try:
            return datetime(cleaned_info[0], cleaned_info[1], cleaned_info[2]).strftime(self.date_format)
        except:
            return None

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
        for item in req:
            # Create an item_dict the dict
            item_dict = dict()
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

    def openi_harvest(self, bounds_list, joined_url, to_harvest, download_no):
        """

        :param bounds_list:
        :param joined_url:
        :param bound:
        :param download_no:
        :return:
        """
        # Initialize
        c = 1
        harvested_data = list()

        # Print Header
        header("Downloading Records... ")

        # Print updates
        if self.verbose:
            print("\nNumber of Records to Download: {0} (maximum block size: {1} rows).".format(
                '{:,.0f}'.format(download_no), str(self.req_limit)))

        for bound in tqdm(bounds_list):
            if c % self.sleep_mini[0] == 0:
                sleep(abs(self.sleep_mini[1] + np.random.normal()))
            elif c % self.sleep_main[0] == 0: # ToDo: main sleep not working...
                if self.verbose:
                    print("Sleeping for %s seconds..." % self.sleep_main[1])
                sleep(abs(self.sleep_main[1] + np.random.normal()))

            # Harvest
            harvested_data += self.openi_block_harvest(joined_url, bound, to_harvest)

            # Update counter
            c += 1

        # Return
        return harvested_data

    def openi_kinesin(self, search_query, to_harvest, total):
        """

        'Walk' along the search query and harvest the data.

        :param search_query:
        :param total:
        :return:
        """
        # Get a list of lists with the bounds
        bounds, download_no = self.openi_bounds(total)

        # Compute a list of search ranges to pass to the Open-i API
        bounds_list = self.openi_bounds_formatter(bounds)

        # Learn the results returned by the API
        to_harvest = self.harvest_vect(to_harvest)

        # Harvest the data
        harvest = self.openi_harvest(bounds_list, search_query, to_harvest, download_no)

        # Convert to a DataFrame and Return
        return pd.DataFrame(harvest).fillna(np.NaN)


# ---------------------------------------------------------------------------------------------
# Image Harvesting
# ---------------------------------------------------------------------------------------------


class _OpeniImages(object):
    """

    """


    def __init__(self, sleep_time, image_save_location, verbose):
        """


        :param sleep_time: suggested: 5.5
        :param image_save_location: suggested: created_img_dirs['raw'])
        :param verbose:
        """
        self.sleep_time = sleep_time
        self.image_save_location = image_save_location
        self.verbose = verbose
        self.assumed_img_format = 'png' # ToDo: a find better solution.

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
        :param img_type: e.g., 'L' or 'G150'.
        :type img_type: ``str``
        :param img_name:
        :type img_name: ``str``
        :return:
        """
        raw_img_name_cln = cln(img_name, extent=2)

        # Get File format
        img_name_format = raw_img_name_cln.split(".")[-1]

        # Extract the name
        image_name_cleaned = raw_img_name_cln[:-(len(img_name_format) + 1)]#.replace(".{0}".format(img_name_format), "")

        # Generate the name
        new_name = "__".join(map(str, [number, image_name_cleaned, img_type]))

        # Return
        return "{0}.{1}".format(new_name, img_name_format)

    def img_harvest(self, img_title, image_web_address):
        """

        Harvest images from a URL and save to disk.

        :param image_web_address:
        :param lag:
        :return:
        """
        # Init
        page = None

        # Define the save path
        image_save_path = os.path.join(self.image_save_location, img_title)

        # Check if the file already exists; if not, download and save it.
        if not os.path.isfile(image_save_path):
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

        # Log of Names
        img_title_log = list()

        # Get the abbreviation for the image type being downloaded
        img_type = self.img_name_abbrev(data_frame)[image_column]

        # Download the images
        header("Downloading Images... ")
        for img_address in tqdm(data_frame[image_column]):
            # Name the image
            img_title = self.img_titler(number=1, img_name=img_address.split("/")[-1], img_type=img_type)

            # Try download and log whether or not Download was sucessful.
            pull_success = self.img_harvest(img_title, img_address)
            result_log[img_address] = pull_success

            # ToDo: make more robust (?) (or simply use result_log keys).
            if pull_success:
                img_title_log.append(img_title)
            else:
                img_title_log.append(None)

        if self.verbose:
            failed_downloads = {k: v for k, v in result_log.items() if v is False}
            if len(failed_downloads.values()):
                header("Failed Downloads: ")
                for k in failed_downloads.keys():
                    print(" - " + k.split("/")[-1])
            else:
                print("\nAll Images Sucessfully Extracted.")

        # Map record of download sucess to img_address (URLs).
        sucesses_log = data_frame[image_column].map(lambda x: result_log.get(x, None) if pd.notnull(x) else None)

        # Return sucesses log
        return sucesses_log.rename("extracted"), img_title_log


# ---------------------------------------------------------------------------------------------
# Construct Database
# ---------------------------------------------------------------------------------------------


class OpenInterface(object):
    """

    Python Interface for the NIH's Open-i API.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
                       Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param download_limit: max. number of results to download.
                           If ``None``, no limit will be imposed (not recommended). Defaults to 60.
    :type download_limit: ``int``
    :param img_sleep_time: time to sleep (in seconds) between requests for images. Noise is added on each call
                           by adding a value from a normal distrubition (with mean = 0, sd = 1). Defaults to 5 seconds.
    :type img_sleep_time: ``int`` or ``float``
    :param date_format: Defaults to '%d/%m/%Y'.
    :type date_format: ``str``
    :param records_sleep_mini: Tuple of the form: (every x downloads, short peroid of time [seconds]). Defaults to (2, 5).
    :type records_sleep_mini: ``tuple``
    :param records_sleep_main: Tuple of the form: (every x downloads, long peroid of time [seconds]). Defaults to (50, 300)
    :type records_sleep_main: ``tuple``
    :param verbose: print additional details.
    :type verbose: ``bool``
    """

    def __init__(self
                 , cache_path=None
                 , download_limit=60
                 , img_sleep_time=5
                 , date_format='%d/%m/%Y'
                 , records_sleep_mini=(2, 5)
                 , records_sleep_main=(50, 300)
                 , verbose=True):
        """

        Initialize the ``OpenInterface()`` Class.

        """
        self._verbose = verbose
        self._root_url = 'https://openi.nlm.nih.gov'

        # Generate Required Caches
        pcc = _package_cache_creator(sub_dir='images', cache_path=cache_path, to_create=['raw', 'processed'])
        self.root_path, self._created_img_dirs = pcc

        pcc2 = _package_cache_creator(sub_dir='search', cache_path=cache_path, to_create=['search_databases_images'])[1]
        self.search_cache_path = pcc2['search_databases_images']

        # Create an instance of the _OpeniRecords() Class
        self._OpeniRecords = _OpeniRecords(root_url=self._root_url
                                           , date_format=date_format
                                           , download_limit=download_limit
                                           , sleep_mini=records_sleep_mini
                                           , sleep_main=records_sleep_main
                                           , verbose=verbose)

        # Create an instance of the `OpeniImages()` Class
        self._OpeniImages = _OpeniImages(sleep_time=img_sleep_time
                                         , image_save_location=self._created_img_dirs['raw']
                                         , verbose=verbose)

        # Permanent record of images in the raw image cache
        self.image_record_database = None
        self._image_record_database_path = os.path.join(self._created_img_dirs['raw'], "image_record_database.p")

        if os.path.isfile(self._image_record_database_path):
            # Read in the existing database
            self.image_record_database = pd.read_pickle(self._image_record_database_path)
        elif not os.path.isfile(self._image_record_database_path):
            self.image_record_database = None

        # Define the current search term
        self.current_search = None
        self.current_search_url = None
        self.current_search_total = None
        self.current_search_database = None
        self._current_search_to_harvest = None

    def _post_processing_text(self, data_frame):
        """

        :param data_frame:
        :return:
        """
        # snake_case from camelCase and lower. ToDo: move out of this method when `_post_processing_img()` is created.
        data_frame.columns = list(map(lambda x: camel_to_snake_case(x).replace("me_sh", "mesh"), data_frame.columns))

        # Run Feature Extracting Tool and Join with `data_frame`.
        pp = pd.DataFrame(data_frame.apply(feature_extract, axis=1).tolist()).fillna(np.NaN)
        data_frame = data_frame.join(pp, how='left')

        # Make the type of Imaging technology type human-readable. ToDo: apply to the other image_modality.
        data_frame['image_modality_major'] = data_frame['image_modality_major'].map(
            lambda x: openi_image_type_params.get(cln(x).lower(), x), na_action='ignore'
        )

        # Look up the article type
        data_frame['article_type'] = data_frame['article_type'].map(
            lambda x: openi_article_type_params.get(cln(x).lower(), x), na_action='ignore'
        )

        return data_frame

    def _openi_search_special_case(self, search_param, blocked, passed):
        """

        :param search_param: one of 'video', 'image_type'...
        :param blocked: muturally exclusive (i.e., all these items cannot be passed together).
        :param passed: values actually passed to `search_param`.
        :return:
        """
        if all(b in passed for b in blocked):
            raise ValueError("`%s` can only contain one of:\n%s" % (search_param, list_to_bulletpoints(blocked)))

    def _openi_search_check(self, search_arguments, search_dict):
        """

        Method to check for invalid search requests.

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
                if k == 'rankby':
                    self._openi_search_special_case(k, blocked=['newest', 'oldest'], passed=v)

    def _exclusions_img_type_merge(self, args, exclusions):
        """

        Merge Image type with Exclusions.

        :param args:
        :param exclusions:
        :return:
        """
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

    def _search_url_formatter(self, api_search_transform, ordered_params):
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

        return "{0}/retrieve.php{1}".format(self._root_url, search_term)

    def _search_probe(self, search_query, print_results):
        """

        :param search_query:
        :type search_query: ``str``
        :param print_results: print the number of results found
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

    def options(self, search_parameter, print_options=True):
        """

        Options for parameters of `openi_search()`.

        :param search_parameter: one of: 'image_type', 'rankby', 'subset', 'collection', 'fields',
                                         'specialties', 'video' or `exclusions`.
        :param print: if True, pretty print the options, else return as a ``list``.
        :return: a list of valid values for a given search `search_parameter`.
        :rtype: ``list``
        """
        # Terms to blocked from displaying to users if search_parameter != 'exclusions'
        exclusions = ['exclude_graphics', 'exclude_multipanel']

        if search_parameter == 'exclusions':
            opts = [i.split("_")[1] for i in exclusions]
        else:
            # Get the relevant dict of params
            search_dict = openi_search_information()[0].get(cln(search_parameter).strip().lower(), None)

            # Report invalid `search_parameter`
            if search_dict is None:
                raise ValueError("'{0}' is not a valid parameter to pass to the Open-i API".format(search_parameter))

            # Remove exclusions term
            opts = [i for i in search_dict[1].keys() if i not in exclusions]
            if not len(opts):
                raise ValueError("Relevant options for '{0}'.".format(search_parameter))

        # Print or Return
        if print_options:
            print(list_to_bulletpoints(opts))
        else:
            return opts

    def _search_clean(self, k, v):
        """

        Define a tool to clean the search terms in `OpenInterface().search()`.

        :param k:
        :param v:
        :return:
        """
        return [cln(i).replace(' ', '_').lower() for i in v] if k != 'query' and v is not None else v

    def _api_url_terms(self, k, v, search_dict):
        """

        Convert values passed into a form the API will understand.

        :param k:
        :param v:
        :param search_dict:
        :return:
        """
        return ','.join([search_dict[k][1][i] for i in v]) if k != 'query' else v

    def _api_url_param(self, x, search_dict):
        """

        Convert param names into a form the API will understand.

        :param x:
        :param search_dict:
        :return:
        """
        return search_dict[x][0] if x != 'query' else 'query'

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
        :param image_type: see `OpenInterface().options('image_type')` for valid values.
        :type image_type: ``list``, ``tuple`` or ``None``.
        :param rankby: see `OpenInterface().options('rankby')` for valid values.
        :type rankby: ``list``, ``tuple`` or ``None``.
        :param subset: see `OpenInterface().options('subset')` for valid values.
        :type subset: ``list``, ``tuple`` or ``None``.
        :param collection: see `OpenInterface().options('collection')` for valid values.
        :type collection: ``list``, ``tuple`` or ``None``.
        :param fields: see `OpenInterface().options('fields')` for valid values.
        :type fields: ``list``, ``tuple`` or ``None``.
        :param specialties: see `OpenInterface().options('specialties')` for valid values.
        :type specialties: ``list``, ``tuple`` or ``None``.
        :param video: see `OpenInterface().options('video')` for valid values. Defaults to ``None``.
        :type video: ``list``, ``tuple`` or ``None``.
        :param exclusions: one or both of: 'graphics', 'multipanel'. Defaults to ['graphics'].      -- Working?
        :type exclusions: ``list``, ``tuple`` or ``None``
        :param print_results: if ``True``, print the number of search results.
        :type print_results: ``bool``
        :return: search URL for the Open-i API.
        :rtype: ``str``
        """
        # Remove 'self' and 'print_results' from locals
        args_cleaned = {k: v for k, v in deepcopy(locals()).items() if k not in ['self', 'print_results']}

        # Block blank searches
        if not len(list(filter(None, args_cleaned.values()))):
            raise ValueError("No Search Criterion Detected. Please specify criterion to narrow your search.")

        # Extract the function arguments
        args = {k: v for k, v in args_cleaned.items() if k != 'exclusions'}

        # Merge `image_type` with `exclusions`
        args = self._exclusions_img_type_merge(args, exclusions)

        # Get the arguments
        search_arguments = {k: '' if k == 'query' and v is None else self._search_clean(k, v) for k, v in args.items()}

        # Save search query
        self.current_search = {k: v for k, v in deepcopy(search_arguments).items() if v is not None}

        # Get Open-i search params
        search_dict, ordered_params = openi_search_information()

        # Add check for all search terms
        self._openi_search_check(search_arguments, search_dict)

        # Perform transformation
        api_search_transform = {self._api_url_param(k, search_dict): self._api_url_terms(k, v, search_dict)
                                for k, v in search_arguments.items() if v is not None}

        # Format `api_search_transform`
        formatted_search = self._search_url_formatter(api_search_transform, ordered_params)

        # Save `formatted_search`
        self.current_search_url = formatted_search
        self.current_search_total, self._current_search_to_harvest = self._search_probe(formatted_search, print_results)

    def _cache_method_checker(self, database_name, action):
        """

        This method checks for some forms of invalid requests to
        the ``OpenInterface().cache()`` method.

        :param database_name: see ``OpenInterface().cache()``.
        :type database_name: ``str`` or ``None``
        :param action: see ``OpenInterface().cache()``.
        :type action: ``str`` or ``None``
        :return: a list of databases found in ``self.search_cache_path``
        :rtype: ``list``
        """
        if self.current_search_database is None and action == 'save':
            raise AttributeError("A dataframe has not yet been harvested using `pull()`.")

        if action not in ['save', 'load', '!DELETE!', None]:
            raise ValueError("`action` must be one of: 'save', 'load', '!DELETE!' or `None`.")

        # Find databases
        databases_found = [f for f in os.listdir(self.search_cache_path) if f.endswith(".p")]

        # Raise if no databases found and action is not 'save'.
        if not len(databases_found) and (action == 'load' or action == '!DELETE!'):
            raise FileNotFoundError("No databases currently cached.")

        # Get files current cached
        if database_name is None and action is not None:
            raise ValueError("if `database_name` is None, `action` must also be None")
        if isinstance(database_name, str) and action is None:
            raise ValueError("`action` cannot be None if `database` is not None")

        return databases_found

    def _cache_method_db_support(self, database_name, action):
        """

        Save, load and destroy support information for a given database.

        :param database_name: see ``OpenInterface().cache()``.
        :type database_name: ``str`` or ``None``
        :param action: see ``OpenInterface().cache()``.
        :type action: ``str`` or ``None``
        """
        # Define the name of the support object
        db_support_name = os.path.join(self.search_cache_path, "{0}_support.p".format(database_name))

        if action == 'save':
            if os.path.isfile(db_support_name):
                raise AttributeError("A support database file named '{0}' already exists in:\n '{1}'\n.".format(
                    db_support_name, self.search_cache_path))
            # Add to Dict
            save_dict = {
                "current_search": self.current_search,
                "current_search_url": self.current_search_url,
                "current_search_total": self.current_search_total,
                "_current_search_to_harvest": self._current_search_to_harvest
            }
            # Save
            pickle.dump(save_dict, open(db_support_name, "wb"))
        elif action == 'load':
            # Load cached support dict
            recovered_db_support_data = pickle.load(open(db_support_name, "rb"))
            # Restore data
            self.current_search = recovered_db_support_data['current_search']
            self.current_search_url = recovered_db_support_data['current_search_url']
            self.current_search_total = recovered_db_support_data['current_search_total']
            self._current_search_to_harvest = recovered_db_support_data['_current_search_to_harvest']
        elif action == '!DELETE!':
            os.remove(db_support_name)

    def _cache_summary(self, databases_found, return_request):
        """

        :param databases_found:
        :param return_request:
        :return:
        """
        if not len(databases_found):
            raise FileNotFoundError("No databases currently cached.")
        if return_request:
            return [i for i in databases_found if not i.endswith("_support.p")]
        else:
            header("Cached Databases: ", flank=False)
            print(list_to_bulletpoints([i.replace(".p", "") for i in databases_found if not i.endswith("_support.p")]))
            return None
        
    def _cache_db_does_not_exist(self, database_name, action, db_path):
        """
        

        :param database_name: 
        :type database_name:
        :param action:
        :type action:
        :param db_path:
        :type db_path:
        :return:
        """
        if action == 'save':
            self.current_search_database.to_pickle(db_path)
            self._cache_method_db_support(database_name, action)
        elif action == 'load' or action == '!DELETE!':
            raise FileNotFoundError("Could not find a database entitled '{0}' in:\n '{1}'.".format(
                database_name, self.search_cache_path))

    def _cache_db_does_exist(self, database_name, action, db_path, return_request):
        """

        :param database_name:
        :type database_name:
        :param action:
        :type action:
        :param db_path:
        :type db_path:
        :param return_request:
        :type return_request:
        :return: `self.current_search_database` if `return_request` is `True`.
        :rtype: `self.current_search_database` or `None`.
        """
        if action == 'save':
            raise AttributeError("A database named '{0}' already exists in:\n '{1}'.".format(
                database_name, self.search_cache_path))
        elif action == 'load':
            self.current_search_database = pd.read_pickle(db_path)
            self._cache_method_db_support(database_name, action)
            if return_request:
                return self.current_search_database
        elif action == '!DELETE!':
            os.remove(db_path)
            self._cache_method_db_support(database_name, action)
            if self._verbose:
                warn("\nThe '{0}' database was successfully deleted from:\n '{1}'.".format(
                    database_name, self.search_cache_path))

    def cache(self, database_name=None, action=None, return_request=True):
        """

        Cache a database, load a cached database to ``self.current_search_database`` or delete a database.

        :param database_name: if `action` is 'save': the name for the database to be saved.
                              if `action` is 'load': the name of the database to be loaded.
                              if `action` is '!DELETE!': the database to delete.
                              if `database_name` is ``None``, a list of current saved database will be provided.
                              Defaults to ``None``.
        :type database_name: ``str`` or ``None``
        :param action: 'save' to cache the current database.
                       'load' to retore an existing database.
                       '!DELETE!' to delete an existing database.
                       Defaults to ``None``.
        :type action: ``str`` or ``None``
        :param return_request:  if `database_name` is None and `return_request` is ``True``, return a list of databases
                                currently cached, else pretty print the list.
                                if `action` is 'load' and `return_request` is ``True``, ``self.current_search_database``
                                AND return the database. Conversely, if `return_request` is ``False``, the database
                                will simply be loaded to ``self.current_search_database``.
        :type return_request: ``bool``
        :return: list of currently cached databases or a cached DataFrame.
        :rtype: ``list``, ``Pandas DataFrame`` or ``None``
        """
        # Check for (some) forms of invalid requests.
        databases_found = self._cache_method_checker(database_name, action)

        if all(x == None or x is None for x in [database_name, action]):
            return self._cache_summary(databases_found, return_request)

        # Path to the database
        db_path = "{0}.p".format(os.path.join(self.search_cache_path, database_name.replace(os.sep, "")))

        if not os.path.isfile(db_path):
            self._cache_db_does_not_exist(database_name, action, db_path)
        elif os.path.isfile(db_path):
            return self._cache_db_does_exist(database_name, action, db_path, return_request)

    def _query_id(self, query_col):
        """

        Classify the dicts in self.image_record_database['query'] in
        `self._image_cache_record_management()`

        :param data_frame:
        :return:
        """
        # Get the unique dicts.
        # The elements in `udicts` are unique.
        # Therefore, its index (+1) provides an axis for classifying other dicts.
        u_dicts = unique_dics(query_col)

        ids = list()
        for q in query_col.tolist():
            for i, u in enumerate(u_dicts, start=1):
                if same_dict(q, u):
                    ids.append(i)

        return ids

    def _img_relation_map(self, data_frame):
        """

        Algorithm to find the index of rows which reference
        the same image in the cache.

        :param data_frame:
        :return:
        """
        # Copy the data_frame
        df = data_frame.copy()

        # Reset the index
        df = df.reset_index(drop=True)

        # Get duplicated img_cache_name occurences
        duplicated_img_refs = (k for k, v in Counter(df['img_cache_name']).items() if v > 1)

        # Get the indices of duplicates
        dup_index = {k: df[df['img_cache_name'] == k].index.tolist() for k in duplicated_img_refs}

        # Create a column of the index (DataFrame.apply() cannot gain access to the index).
        df['index_temp'] = df.index

        def related(x):
            """Function to look for references to the same image in the cache"""
            if x['img_cache_name'] in dup_index:
                return tuple(sorted([i for i in dup_index[x['img_cache_name']] if i != x['index_temp']]))
            else:
                return np.NaN

        # Apply `relate()`
        df['shared_img_ref'] = df.apply(related, axis=1)

        # Delete temp_index
        del df['index_temp']

        return df

    def _image_cache_record_relationships(self, data_frame, action='both'):
        """

        :param data_frame:
        :return:
        """
        # Get image mappings
        if action in ['img_rel', 'both']:
            data_frame = self._img_relation_map(data_frame)

        # Compute an ID for each query
        if action in ['q_uid', 'both']:
            data_frame['query_uid'] = self._query_id(data_frame['query'])

        return data_frame

    def _image_cache_record_management(self):
        """

        Maintain Record of files in the image cache.
        
        """
        temp_df = None
        if not os.path.isfile(self._image_record_database_path):
            # Then the image_record_database == current_search_database
            self.image_record_database = self.current_search_database
            # Add the Search Term
            self.image_record_database['query'] = [self.current_search] * self.image_record_database.shape[0]
            # Map Relationships
            self.image_record_database = self._image_cache_record_relationships(self.image_record_database)
            # Save to disk
            self.image_record_database.to_pickle(self._image_record_database_path)
        elif os.path.isfile(self._image_record_database_path):
            # Read in the existing database
            self.image_record_database = pd.read_pickle(self._image_record_database_path)
            # Combine with the current search, if not None
            if self.current_search_database is not None:
                # Create a copy of self.current_search_database to operate on.
                temp_df = self.current_search_database.copy()
                # Add the Search Term to the temporary dataframe
                temp_df['query'] = [self.current_search] * temp_df.shape[0]
                # Append the current search database to the existing database
                self.image_record_database = self.image_record_database.append(temp_df)
                # Name search ids their unqiue id. Shields distinct search that yeilded the same image frop dropping.
                self.image_record_database = self._image_cache_record_relationships(self.image_record_database, 'q_uid')
                # Drop Duplicates, favoring the first instance of a row
                self.image_record_database = self.image_record_database.drop_duplicates(
                    subset=hashable_cols(self.image_record_database), keep='first')
                # Map Relationships between rows in the df.
                self.image_record_database = self._image_cache_record_relationships(self.image_record_database, 'both')
                # Save back out to disk
                self.image_record_database.to_pickle(self._image_record_database_path)
                # Delete temp_df
                del temp_df

    def _pull_search_data(self, new_records_pull):
        """

        Define or evolve `search_data`.

        :param new_records_pull: see ``OpenInterface().pull()``.
        :type new_records_pull: ``bool``
        :return: `search_data`
        :rtype: ``Pandas DataFrame``
        """
        if new_records_pull is False and self.current_search_database is not None:
            search_data = self.current_search_database
        elif new_records_pull is False and self.current_search_database is None:
            raise ValueError("`self.current_search_database` cannot be None if `new_records_pull` is `False`.")
        else:
            self.current_search_database = None
            search_data = self._OpeniRecords.openi_kinesin(self.current_search_url
                                                           , to_harvest=self._current_search_to_harvest
                                                           , total=self.current_search_total)
            search_data = self._post_processing_text(search_data)

        return search_data

    def _pull_image_col(self, search_data, image_quality):
        """

        Pull images if `image_quality` is not ``None``, else return `search_data` 'as-is'.

        :param search_data: see ``OpenInterface().pull()``.
        :param search_data: ``Pandas DataFrame``
        :param image_quality: see ``OpenInterface().pull()``.
        :type image_quality: ``str`` or ``None``
        :return: `search_data` with columns on th effort to pull images
               if `image_quality` is not ``None`` else ``search_data``.
        :rtype: ``Pandas DataFrame``
        """
        if image_quality is not None:
            image_col = "img_{0}".format(image_quality)
            search_data['img_extracted'], search_data['img_cache_name'] = self._OpeniImages.bulk_img_harvest(
                search_data, image_col
            )
        elif self._verbose:
            warn("\nNo attempt was made to download images because `image_quality` is `None`.")

        return search_data

    def _pull_search_wrapper(self, image_quality, new_records_pull):
        """

        Wrapper for ``OpenInterface()._pull_search_data()`` and ``OpenInterface()._pull_image_col()``.

        :param new_records_pull: see ``OpenInterface().pull()``.
        :type new_records_pull: ``bool``
        :param image_quality: see ``OpenInterface().pull()``.
        :type image_quality: ``str`` or ``None``
        :return: the search DataFrame
        :rtype: ``Pandas DataFrame``
        """
        # Get database for the search
        data_pull = self._pull_search_data(new_records_pull)

        # Harvest images (if requested) and Return
        data_pull_img = self._pull_image_col(data_pull, image_quality)

        # Update `self.current_search_database`.
        self.current_search_database = data_pull_img

        return data_pull_img

    def pull(self, image_quality='large', new_records_pull=True):
        """

        Pull (i.e., download) the current search.

        :param image_quality: one of: 'large', 'grid150', 'thumb', 'thumb_large' or ``None``. Defaults to 'large'.
                          If ``None``, no attempt will be made to download images.
        :type image_quality: ``str`` or ``None``
        :param new_records_pull: if True, download the data for the current search.
                                 if False, use self.current_search_database. This can be useful
                                 if one wishes to initially set `image_quality` to `None`,
                                 truncate or otherwise modify `self.current_search_database` and then
                                 download images.
        :type new_records_pull: ``bool``
        :return: a DataFrame with the record information.
                 If `image_quality` is not None, images will also be harvested and cached.
        :rtype: ``Pandas DataFrame``
        """
        if self.current_search_url is None:
            raise ValueError("A search has not been defined. Please call `OpenInterface().search()`.")

        # Define allowed image types to pull
        allowed_image_quality_types = ('large', 'grid150', 'thumb', 'thumb_large')

        # Check `image_quality` is valid
        if image_quality is not None and image_quality not in allowed_image_quality_types:
            raise ValueError("`image_quality` must be `None` or one of:\n{0}.".format(
                list_to_bulletpoints(allowed_image_quality_types)))

        # Define or evolve `search_data` and Download Images (if requested).
        search_data = self._pull_search_wrapper(image_quality, new_records_pull)

        # Update `self.image_record_database`
        if image_quality is not None:
            self._image_cache_record_management()

        return search_data






































