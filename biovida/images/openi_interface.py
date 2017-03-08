"""

    Open-i Interface
    ~~~~~~~~~~~~~~~~

"""
# Imports
import re
import os
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import floor
from copy import deepcopy
from datetime import datetime

from biovida import __version__

# General Image Support Tools
from biovida.images._image_tools import TIME_FORMAT
from biovida.images._image_tools import NoResultsFound
from biovida.images._image_tools import resetting_label
from biovida.images._image_tools import sleep_with_noise

# Database Management
from biovida.images.image_cache_mgmt import _load_temp_dbs
from biovida.images.image_cache_mgmt import _records_db_merge
from biovida.images.image_cache_mgmt import _openi_image_relation_map
from biovida.images.image_cache_mgmt import _record_update_dbs_joiner
from biovida.images.image_cache_mgmt import _prune_rows_with_deleted_images

# Open-i Support tools
from biovida.images._interface_support.openi.openi_support_tools import iter_join
from biovida.images._interface_support.openi.openi_support_tools import url_combine
from biovida.images._interface_support.openi.openi_support_tools import null_convert
from biovida.images._interface_support.openi.openi_support_tools import ImageProblemBasedOnText

# Open-i API Parameters Information
from biovida.images._interface_support.openi.openi_parameters import openi_search_information

# Open-i Raw Text Processing
from biovida.images._interface_support.openi.openi_text_processing import openi_raw_extract_and_clean

# Cache Management
from biovida.support_tools._cache_management import package_cache_creator

# General Support Tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import camel_to_snake_case
from biovida.support_tools.support_tools import list_to_bulletpoints

# Start tqdm
tqdm.pandas(desc='status')


# ----------------------------------------------------------------------------------------------------------
# Searching
# ----------------------------------------------------------------------------------------------------------


class _OpeniSearch(object):
    """

    """

    def __init__(self):
        self._root_url = 'https://openi.nlm.nih.gov'

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
        general_error_msg = "'{0}' is not valid for `{1}`.\nValid values for `{1}`:\n{2}"

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
            raise NoResultsFound("\n\nPlease Try Refining Your Search.")

        # Print number of results found
        if print_results:
            print("\nResults Found: %s." % ('{:,.0f}'.format(total)))

        return total, sample['list'][0]

    def options(self, search_parameter, print_options=True):
        """

        Options for parameters of `openi_search()`.

        :param search_parameter: one of: 'image_type', 'rankby', 'article_type', 'subset', 'collection', 'fields',
                                         'specialties', 'video' or `exclusions`.
        :param print_options: if True, pretty print the options, else return as a ``list``.
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

        Define a tool to clean the search terms in `OpeniInterface().search()`.

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

    def search(self,
               query,
               image_type=None,
               rankby=None,
               article_type=None,
               subset=None,
               collection=None,
               fields=None,
               specialties=None,
               video=None,
               exclusions=None,
               print_results=True):
        """

        Tool to generate a search term (URL) for the NIH's Open-i API.
        The computed term is stored as a class attribute (``INSTANCE.current_search_url``)

        :param query: a search term. ``None`` will be converted to an empty string.
        :type query: ``str`` or ``None``
        :param image_type: see ``OpeniInterface().options('image_type')`` for valid values.
        :type image_type: ``str``, ``list``, ``tuple`` or ``None``
        :param rankby: see ``OpeniInterface().options('rankby')`` for valid values.
        :type rankby: ``str``, ``list``, ``tuple`` or ``None``
        :param article_type: see ``OpeniInterface().options('article_type')`` for valid values.
        :type article_type: ``str``, ``list``, ``tuple`` or ``None``
        :param subset: see ``OpeniInterface().options('subset')`` for valid values.
        :type subset: ``str``, ``list``, ``tuple`` or ``None``
        :param collection: see ``OpeniInterface().options('collection')`` for valid values.
        :type collection: ``str``, ``list``, ``tuple`` or ``None``
        :param fields: see ``OpeniInterface().options('fields')`` for valid values.
        :type fields: ``str``, ``list``, ``tuple`` or ``None``
        :param specialties: see ``OpeniInterface().options('specialties')`` for valid values.
        :type specialties: ``str``, ``list``, ``tuple`` or ``None``
        :param video: see ``OpeniInterface().options('video')`` for valid values. Defaults to ``None``.
        :type video: ``str``, ``list``, ``tuple`` or ``None``
        :param exclusions: one or both of: 'graphics', 'multipanel'. See: ``OpeniInterface.search()``.
        :type exclusions: ``list``, ``tuple`` or ``None``
        :param print_results: if ``True``, print the number of search results.
        :type print_results: ``bool``
        """
        # Remove 'self' and 'print_results' from locals
        args_cleaned = {k: v for k, v in deepcopy(locals()).items() if k not in ['self', 'print_results']}

        # Block blank searches
        if not len(list(filter(None, args_cleaned.values()))):
            raise ValueError("No Search Criterion Detected. Please specify criterion to narrow your search.")

        # Extract the function arguments and format values.
        args = {k: [v] if isinstance(v, str) and k != 'query' else v for k, v in args_cleaned.items() if k != 'exclusions'}

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
        search_url = self._search_url_formatter(api_search_transform, ordered_params)

        # Unpack the probe containing information on the total number of results and list of results to harvest
        current_search_total, current_search_to_harvest = self._search_probe(search_url, print_results)

        # Return
        return {"query": args_cleaned,
                "search_url": search_url,
                "current_search_total": current_search_total,
                "current_search_to_harvest": current_search_to_harvest}


# ----------------------------------------------------------------------------------------------------------
# Pull Records from the NIH's Open-i API
# ----------------------------------------------------------------------------------------------------------


class _OpeniRecords(object):
    """

    Tools to Pull Records from the Open-i API.

    """

    def __init__(self, root_url, date_format, verbose, cache_path, req_limit=30):
        """

        :param root_url: suggested: 'https://openi.nlm.nih.gov'
        :type root_url: ``str``
        :param date_format: suggested: "%d/%m/%Y" (consider leaving as datetime)
        :type date_format: ``str``
        :param verbose: print additional details.
        :type verbose: ``bool``
        :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
        :type cache_path: ``str``
        :param req_limit: Defaults to 30 (max allowed by Open-i; see: https://openi.nlm.nih.gov/services.php?it=xg).
        :type req_limit: ``int``
        """
        self.root_url = root_url
        self.date_format = date_format
        self._verbose = verbose
        self._cache_path = cache_path
        self.req_limit = req_limit

        self.records_db = None
        self.download_limit = 60  # set to reasonable default.

        # Sleep Time
        self.records_sleep_time = None

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
            info = [int(i) if i.isdigit() else None for i in
                    [date_dict['year'], date_dict['month'], date_dict['day']]]
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
            elif isinstance(v, dict) and any(
                            dmy in map(lambda x: x.lower(), v.keys()) for dmy in ['day', 'month', 'year']):
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

        root_url_columns = ('detailed_query_url', 'get_article_figures', 'similar_in_collection', 'similar_in_results')
        def append_root_url(item):
            """Check whether or not to add `self.root_url` to a column."""
            return 'img_' in item or any(c == item for c in root_url_columns)

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
                elif append_root_url(camel_to_snake_case(j)):
                    item_dict[j] = url_combine(self.root_url, null_convert(item.get(j, None)))
                else:
                    item_dict[j] = null_convert(item.get(j, None))

            list_of_dicts.append(item_dict)

        return list_of_dicts

    def openi_harvest(self, bounds_list, joined_url, to_harvest, records_sleep_time, download_no):
        """


        :param bounds_list:
        :param joined_url:
        :param to_harvest:
        :param records_sleep_time:
        :param download_no:
        :return:
        """
        # Initialize
        harvested_data = list()
        header("Downloading Records... ")

        if self._verbose:
            print("\nNumber of Records to Download: {0} (chunk size: {1} records).".format(
                '{:,.0f}'.format(download_no), str(self.req_limit)))

        for c, bound in enumerate(tqdm(bounds_list), start=1):
            if c % records_sleep_time[0] == 0:
                sleep_with_noise(amount_of_time=records_sleep_time[1])

            # Harvest
            harvested_data += self.openi_block_harvest(joined_url, bound, to_harvest)

        return harvested_data

    def _apply_clinical_case_only(self, records_db):
        """

        Remove records (dataframe rows) which are not of clinical encounters.

        Note: this is here, and not in ``_OpeniImages().pull_images()`` b/c the
        Open-i API does not have an 'encounter' param (which it probably should).

        :param records_db: the ``records_db`` as evolved in ``records_pull()``.
        :type records_db: ``Pandas DataFrame``
        :return: see description.
        :rtype: ``Pandas DataFrame``
        """
        clinical_article_types = ('encounter', 'case_report')
        def test(article_type):
            if isinstance(article_type, str) and article_type in clinical_article_types:
                return True
            else:
                return False

        records_db = records_db[records_db['article_type'].map(test)].reset_index(drop=True)
        if records_db.shape[0] == 0:
            raise NoResultsFound("\nNo results remained after the `clinical_cases_only` restriction was applied.")
        return records_db

    def records_pull(self,
                     search_url,
                     to_harvest,
                     total,
                     query,
                     pull_time,
                     records_sleep_time,
                     clinical_cases_only,
                     download_limit=None):
        """

        'Walk' along the search query and harvest the data.

        :param search_url:
        :param to_harvest:
        :param total:
        :param query:
        :param pull_time:
        :param records_sleep_time: (every x downloads, period of time [seconds])
        :param clinical_cases_only: see ``OpeniInterface().pull()``.
        :type clinical_cases_only: ``bool``
        :param download_limit:
        :return:
        """
        if isinstance(download_limit, int):
            self.download_limit = download_limit

        # Get a list of lists with the bounds
        bounds, download_no = self.openi_bounds(total)

        # Compute a list of search ranges to pass to the Open-i API
        bounds_list = self.openi_bounds_formatter(bounds)

        # Learn the results returned by the API
        to_harvest = self.harvest_vect(to_harvest)

        # Harvest the data
        harvest = self.openi_harvest(bounds_list=bounds_list,
                                     joined_url=search_url,
                                     to_harvest=to_harvest,
                                     records_sleep_time=records_sleep_time,
                                     download_no=download_no)

        # Convert to a DataFrame
        records_db = pd.DataFrame(harvest).fillna(np.NaN)

        # Process Text
        records_db = openi_raw_extract_and_clean(data_frame=records_db,
                                                 verbose=self._verbose,
                                                 cache_path=self._cache_path)

        # Add the query
        records_db['query'] = [query] * records_db.shape[0]
        records_db['pull_time'] = [pull_time] * records_db.shape[0]

        # Add the Version of BioVida which generated the DataFrame
        records_db['biovida_version'] = [__version__] * records_db.shape[0]

        if clinical_cases_only:
            records_db = self._apply_clinical_case_only(records_db)

        # Add to attrs.
        self.records_db = records_db

        return self.records_db


# ----------------------------------------------------------------------------------------------------------
# Image Harvesting
# ----------------------------------------------------------------------------------------------------------


class _OpeniImages(object):
    """

    :param image_save_location: suggested: created_img_dirs['openi'])
    :param database_save_location:
    :param verbose:
    """

    def __init__(self, image_save_location, database_save_location, verbose):
        self.image_save_location = image_save_location
        self._verbose = verbose

        # Database
        self.records_db_images = None
        self.real_time_update_db = None
        self.real_time_update_db_path = None

        self.temp_directory_path = os.path.join(database_save_location, "__temp__")

    def _create_temp_directory_path(self):
        """

        Check if ``self.temp_directory_path`` exists. If not, create it.

        """
        if not os.path.isdir(self.temp_directory_path):
            os.makedirs(self.temp_directory_path)

    def _instantiate_real_time_update_db(self, db_index, pull_time):
        """

        Create the ``real_time_update_db`` and define the path to the location where it will be saved.

        :param db_index: the index of the ``real_time_update_db`` dataframe (should be from ``records_db``).
        :type db_index: ``Pandas Series``
        :param pull_time: see ``pull_images()``
        :type pull_time: ``str``
        """
        # Define the path to save `self.real_time_update_db` to.
        self.real_time_update_db_path = os.path.join(self.temp_directory_path, "{0}__update_db.p".format(pull_time))

        # Define columns
        real_time_update_columns = ['cached_images_path', 'download_success']

        # Instantiate
        db = pd.DataFrame(columns=real_time_update_columns, index=db_index).replace({np.NaN: None})
        self.real_time_update_db = db

    def _save_real_time_update_db(self):
        """

        Save the ``real_time_update_db`` to disk.

        """
        # Save the `real_time_update_db` to disk.
        self.real_time_update_db.to_pickle(self.real_time_update_db_path)

    def _image_titler(self, url, image_size):
        """

        Generate a title for the images.

        :param url:
        :param image_size:
        :return:
        """
        # Get the actual file name
        base = os.path.basename(url)

        # Separate the name from the image type
        bname, image_format = os.path.splitext(base)

        # Generate and clean strings to populate the name format below. Note: 1 = file number
        # (in case medpix has images with multiple segments -- though, it doesn't appear to currently.)
        replacement_terms = map(lambda x: cln(x), (str(1), bname, image_size, image_format.replace(".", "")))

        # Generate the name for the image
        image_name = "{0}__{1}__{2}.{3}".format(*replacement_terms)

        # Return the save path.
        return os.path.join(self.image_save_location, image_name)

    def _individual_image_harvest(self, index, image_url, image_save_path, block):
        """

        Harvests a single image.

        :param index: the row (index) that it currently being on through.
        :param index: ``int``
        :param image_url: URL to the image.
        :type image_url: ``str``
        :param image_save_path: the location to save the image.
        :type image_save_path: ``str``
        :param block: whether or not to block downloading the image if it does not already exist in the cache.
        :type block: ``bool``
        :return: `1` if an image was downloaded, `0` otherwise.
        :rtype: ``int``
        """
        image_downloaded = 0
        try:
            # Only download if the file does not already exist in the cache.
            if not os.path.isfile(image_save_path):
                if block:
                    raise ImageProblemBasedOnText

                # Get the image
                page = requests.get(image_url)
                # Save to disk
                with open(image_save_path, 'wb') as img:
                    img.write(page.content)
                image_downloaded = 1

            self.real_time_update_db.set_value(index, 'cached_images_path', image_save_path)
            self.real_time_update_db.set_value(index, 'download_success', True)
            self._save_real_time_update_db()
        except:
            self.real_time_update_db.set_value(index, 'cached_images_path', np.NaN)
            self.real_time_update_db.set_value(index, 'download_success', False)
            self._save_real_time_update_db()

        return image_downloaded

    def _pull_images_engine(self, harvesting_information, images_sleep_time, image_size, use_image_caption):
        """

        Use ``_individual_image_harvest()`` to download all of the data (images) in ``harvesting_information``.

        :param harvesting_information: as evolved inside ``harvesting_information``
        :type harvesting_information: ``list``
        :param images_sleep_time: see ``pull_images()``
        :type images_sleep_time: ``tuple``
        :param image_size: see ``pull_images()``
        :type image_size: ``str``
        :param use_image_caption: if ``True`` block downloading of an image if its caption suggests the presence
                                  of problematic image properties (e.g., 'arrows') likely to corrupt
                                  a dataset intended for machine learning. Defaults to ``False``.
        :type use_image_caption: ``bool``
        """
        if self._verbose:
            header("Obtaining Images... ")

        def block_decision(ipt):
            """Decide whether or not to block the downloading."""
            return use_image_caption == True and isinstance(ipt, (list, tuple)) and len(ipt)
            
        download_count = 0
        for index, image_url, image_problems_text in tqdm(harvesting_information):
            # Generate the save path for the image
            image_save_path = self._image_titler(url=image_url, image_size=image_size)

            # Save the image
            download_count += self._individual_image_harvest(index=index,
                                                             image_url=image_url,
                                                             image_save_path=image_save_path,
                                                             block=block_decision(image_problems_text))

            # Sleep when `download_count` 'builds up' to images_sleep_time[0].
            if download_count == images_sleep_time[0]:
                sleep_with_noise(amount_of_time=images_sleep_time[1])
                download_count = 0  # reset

    def pull_images(self, records_db, image_size, pull_time, images_sleep_time, use_image_caption):
        """

        Pull images based in ``records_db``.

        :param records_db: yeild of ``_OpeniRecords().records_pull()``
        :type records_db: ``Pandas DataFrame``
        :param image_size: one of 'grid150', 'large', 'thumb' or 'thumb_large'.
        :type image_size: ``str``
        :param pull_time: the time the query was made by ``OpeniInterface.search()``.
        :type pull_time: ``datetime``
        :param images_sleep_time: tuple of the form: ``(every x downloads, period of time [seconds])``. Defaults to ``(10, 1.5)``.
                                   Note: noise is randomly added to the sleep time by sampling from a normal distribution
                                   (with mean = 0, sd = 0.75).
        :type images_sleep_time: ``tuple``
        :param use_image_caption: if ``True`` block downloading of an image if its caption suggests the presence
                                  of problematic image properties (e.g., 'arrows') likely to corrupt
                                  a dataset intended for machine learning. Defaults to ``False``.
        :type use_image_caption: ``bool``
        :return: `records_db` with the addition of `cached_images_path` and `download_success` columns.
        :rtype: ``Pandas DataFrame``
        """
        self._create_temp_directory_path()
        self.records_db_images = records_db.copy(deep=True)
        
        # Save `records_db_images` to the __temp__ folder
        self.records_db_images.to_pickle(os.path.join(self.temp_directory_path, "{0}__records_db.p".format(pull_time)))

        # Instantiate `self.real_time_update_db`
        self._instantiate_real_time_update_db(db_index=self.records_db_images.index, pull_time=pull_time)

        if image_size not in ('grid150', 'large', 'thumb', 'thumb_large'):
            raise ValueError("`image_size` must be one of: 'grid150', 'large', 'thumb' or 'thumb_large'.")
        image_column = "img_{0}".format(image_size)

        # Extract needed information from the `records_db_images` dataframe to loop over.
        harvesting_information = list(zip(*[self.records_db_images.index,
                                            self.records_db_images[image_column],
                                            self.records_db_images['image_problems_from_text']]))

        # Harvest
        self._pull_images_engine(harvesting_information=harvesting_information,
                                 images_sleep_time=images_sleep_time,
                                 image_size=image_size,
                                 use_image_caption=use_image_caption)

        return _record_update_dbs_joiner(records_db=self.records_db_images, update_db=self.real_time_update_db)


# ----------------------------------------------------------------------------------------------------------
# Construct Database
# ----------------------------------------------------------------------------------------------------------


class OpeniInterface(object):
    """

    Python Interface for the NIH's `Open-i <https://openi.nlm.nih.gov>`_ API.

    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    :param verbose: print additional details.
    :type verbose: ``bool``
    """

    def _save_cache_records_db(self):
        """

        Save ``cache_records_db`` to 'disk'.

        """
        self.cache_records_db.to_pickle(self._cache_records_db_save_path)

    def _load_prune_cache_records_db(self, load):
        """

        Load and Prune the ``cache_records_db``.

        :param load: if ``True`` load the ``cache_records_db`` dataframe in from disk.
        :type load: ``bool``
        """
        cache_records_db = pd.read_pickle(self._cache_records_db_save_path) if load else self.cache_records_db
        self.cache_records_db = _prune_rows_with_deleted_images(cache_records_db=cache_records_db,
                                                                columns=['cached_images_path'],
                                                                save_path=self._cache_records_db_save_path)

    def _latent_temp_dir(self):
        """

        Load a '__temp__' folder of image record databases which were not merged (and subsequently destroyed)
        before python exited (the ``pull()`` method, specifically).

        """
        if os.path.isdir(self._Images.temp_directory_path):
            # Load the latent database(s).
            records_db_update = _load_temp_dbs(temp_db_path=self._Images.temp_directory_path)
            if records_db_update is not None:
                # Update `self.current_records_db`.
                self._openi_cache_records_db_handler(current_records_db=self.cache_records_db,
                                                     records_db_update=records_db_update)
                # Delete the latent 'databases/__temp__' folder.
                shutil.rmtree(self._cache_records_db_save_path, ignore_errors=True)

    def _openi_cache_records_db_handler(self, current_records_db, records_db_update):
        """

        1. if cache_records_db.p doesn't exist, simply save ``records_db_update`` to disk.
        2. if cache_records_db.p does exist, merge with ``records_db_update`` and then save to disk.

        :param current_records_db:
        :type current_records_db:
        :param records_db_update:
        :type records_db_update:
        """
        def rows_to_conserve_func(x):
            return x['download_success'] == True

        if current_records_db is None and records_db_update is None:
            raise ValueError("`current_records_db` and `records_db_update` cannot both be None.")
        elif current_records_db is not None and records_db_update is None:
            data_frame = current_records_db
            self.cache_records_db = data_frame[data_frame.apply(rows_to_conserve_func, axis=1)].reset_index(drop=True)
        elif current_records_db is None and records_db_update is not None:
            data_frame = _openi_image_relation_map(records_db_update)
            self.cache_records_db = data_frame[data_frame.apply(rows_to_conserve_func, axis=1)].reset_index(drop=True)
        else:
            duplicates_subset_columns = ['img_grid150', 'img_large', 'img_thumb', 'img_thumb_large',
                                         'query', 'cached_images_path', 'download_success']
            self.cache_records_db = _records_db_merge(interface_name='OpeniInterface',
                                                      current_records_db=current_records_db,
                                                      records_db_update=records_db_update,
                                                      columns_with_dicts=('query', 'parsed_abstract'),
                                                      duplicates_subset_columns=duplicates_subset_columns,
                                                      rows_to_conserve_func=rows_to_conserve_func,
                                                      post_concat_mapping=('uid_instance', 'uid', resetting_label))

        # Save to disk
        self._save_cache_records_db()

    def __init__(self, cache_path=None, verbose=True):
        self._cache_path = cache_path
        self._verbose = verbose
        self._root_url = 'https://openi.nlm.nih.gov'
        self._date_format = '%d/%m/%Y',

        # Generate Required Caches
        _, self._created_img_dirs = package_cache_creator(sub_dir='images',
                                                          cache_path=cache_path,
                                                          to_create=['openi'],
                                                          nest=[('openi', 'aux'), ('openi', 'raw'),
                                                                ('openi', 'databases')],
                                                          requires_medpix_logo=True)

        self._ROOT_PATH = self._created_img_dirs['ROOT_PATH']

        # Instantiate Classes
        self._Search = _OpeniSearch()

        self._Records = _OpeniRecords(root_url=self._root_url,
                                      date_format=self._date_format,
                                      verbose=verbose,
                                      cache_path=cache_path)

        self._Images = _OpeniImages(image_save_location=self._created_img_dirs['raw'],
                                    database_save_location=self._created_img_dirs['databases'],
                                    verbose=verbose)

        # Search attributes
        self._pull_time = None
        self.current_query = None
        self.current_search_url = None
        self.current_search_total = None
        self._current_search_to_harvest = None

        # Databases
        self.records_db = None

        # Path to cache record db
        self._cache_records_db_save_path = os.path.join(self._created_img_dirs['databases'],
                                                        'openi_cache_records_db.p')

        # Load the cache record database, if it exists
        if os.path.isfile(self._cache_records_db_save_path):
            self._load_prune_cache_records_db(load=True)
        else:
            self.cache_records_db = None

        # Load in databases in 'databases/__temp__', if they exist
        self._latent_temp_dir()

    def options(self, search_parameter, print_options=True):
        """

        Options for parameters of ``search()``.

        :param search_parameter: one of: 'image_type', 'rankby', 'article_type', 'subset', 'collection', 'fields',
                                         'specialties', 'video' or 'exclusions'.
        :type search_parameter: ``str``
        :param print_options: if ``True``, pretty print the options, else return as a ``list``. Defaults to ``True``.
        :type print_options: ``bool``
        :return: a list of valid values for a given search ``search_parameter``.
        :rtype: ``list``
        """
        return self._Search.options(search_parameter, print_options)

    def search(self,
               query=None,
               image_type=None,
               rankby=None,
               article_type=None,
               subset=None,
               collection=None,
               fields=None,
               specialties=None,
               video=None,
               exclusions=['graphics'],
               print_results=True):
        """

        Tool to generate a search term (URL) for the NIH's Open-i API.
        The computed term is stored as a class attribute (``INSTANCE.current_search_url``)

        :param query: a search term. ``None`` will be converted to an empty string.
        :type query: ``str`` or ``None``
        :param image_type: see ``OpeniInterface().options('image_type')`` for valid values.
        :type image_type: ``str``, ``list``, ``tuple`` or ``None``
        :param rankby: see ``OpeniInterface().options('rankby')`` for valid values.
        :type rankby: ``str``, ``list``, ``tuple`` or ``None``
        :param article_type: see ``OpeniInterface().options('article_type')`` for valid values. Defaults to 'case_report'.
        :type article_type: ``str``, ``list``, ``tuple`` or ``None``
        :param subset: see ``OpeniInterface().options('subset')`` for valid values.
        :type subset: ``str``, ``list``, ``tuple`` or ``None``
        :param collection: see ``OpeniInterface().options('collection')`` for valid values.
        :type collection: ``str``, ``list``, ``tuple`` or ``None``
        :param fields: see ``OpeniInterface().options('fields')`` for valid values.
        :type fields: ``str``, ``list``, ``tuple`` or ``None``
        :param specialties: see ``OpeniInterface().options('specialties')`` for valid values.
        :type specialties: ``str``, ``list``, ``tuple`` or ``None``
        :param video: see ``OpeniInterface().options('video')`` for valid values.
        :type video: ``str``, ``list``, ``tuple`` or ``None``
        :param exclusions: one or both of: 'graphics', 'multipanel'. Defaults to ``['graphics']``.

                    .. note::

                           Excluding 'multipanel' can result in images that *are* multipanel
                           being returned from Open-i API. For this reason, including 'multipanel'
                           is not currently recommended.

        :type exclusions: ``list``, ``tuple`` or ``None``
        :param print_results: if ``True``, print the number of search results.
        :type print_results: ``bool``

        .. note::

                If passing a single option to ``image_type``, ``rankby``, ``article_type``, ``subset``,
                ``collection``, ``fields``, ``specialties`` or ``video``, a string can be used, e.g.,
                ``...image_type='ct')``. For multiple values, a list or tuple must be used, e.g.,
                ``...image_type=('ct', 'mri')``.

        """
        search = self._Search.search(query=query,
                                     image_type=image_type,
                                     rankby=rankby,
                                     article_type=article_type,
                                     subset=subset,
                                     collection=collection,
                                     fields=fields,
                                     specialties=specialties,
                                     video=video,
                                     exclusions=exclusions,
                                     print_results=print_results)

        # Save the search to the 'outer' class instance
        self.current_query = search['query']
        self.current_search_url = search['search_url']
        self.current_search_total = search['current_search_total']
        self._current_search_to_harvest = search['current_search_to_harvest']

    def pull(self,
             new_records_pull=True,
             image_size='large',
             records_sleep_time=(10, 1.5),
             images_sleep_time=(10, 1.5),
             download_limit=60,
             clinical_cases_only=True,
             use_image_caption=False):
        """

        Pull (i.e., download) the current search.

        In addition to the columns provided by Open-i, this method will automatically generate the
        following columns by analyzing the pulled data:

        - ``'age'``
        - ``'sex'``
        - ``'ethnicity'``
        - ``'diagnosis'``
        - ``'parsed_abstract'``
        - duration of illness (``'illness_duration_years'``)
        - the imaging modality (e.g., MRI) used, based on the text associated with the image (``'imaging_modality_from_text'``)
        - the plane ('axial', 'coronal' or 'sagittal') of the image (``'image_plane'``)
        - image problems ('arrows', 'asterisks' and 'grids') inferred from the image caption (``'image_problems_from_text'``)

        .. note::

            The 'parsed_abstract' column contains abstracts coerced into dictionaries where the subheadings of the abstract
            form the keys and their associated information form the values. For example, a *MedPix* image will typically yield
            a dictionary with the following keys: 'history', 'finding', 'ddx' (differential diagnosis), 'dxhow' and 'exam'.

        .. warning::

            *MedPix* images include a distinct 'diagnosis' section. For images from other sources, the ``'diagnosis'``
            column is obtained by analyzing the text associated with the image. This analysis could produce inaccuracies.

        :param new_records_pull: if ``True``, download the data for the current search. If ``False``, use ``INSTANCE.records_db``.

            .. note::

               Setting ``new_records_pull=False`` can be useful if one wishes to initially set ``image_size=None``,
               truncate or otherwise modify ``INSTANCE.records_db`` and then download images.

        :type new_records_pull: ``bool``
        :param image_size: one of: 'large', 'grid150', 'thumb', 'thumb_large' or ``None``. Defaults to 'large'.
                          If ``None``, no attempt will be made to download images.
        :type image_size: ``str`` or ``None``
        :param records_sleep_time: tuple of the form: ``(every x downloads, period of time [seconds])``. Defaults to ``(10, 1.5)``.
                               Note: noise is randomly added to the sleep time by sampling from a normal distribution
                               (with mean = 0, sd = 0.75).
        :type records_sleep_time: ``tuple``
        :param images_sleep_time: tuple of the form: ``(every x downloads, period of time [seconds])``. Defaults to ``(10, 1.5)``.
                                Note: noise is randomly added to the sleep time by sampling from a normal distribution
                                (with mean = 0, sd = 0.75).
        :type images_sleep_time: ``tuple``
        :param download_limit: max. number of results to download. If ``None``, no limit will be imposed
                              (not recommended). Defaults to 60.
        :type download_limit: ``int``
        :param clinical_cases_only: if ``True`` require that the data harvested is of a clinical case. Specifically,
                                    this parameter requies that 'article_type' is one of: 'encounter', 'case_report'.
                                    Defaults to ``True``.
        :type clinical_cases_only: ``bool``
        :param use_image_caption: if ``True`` block downloading of an image if its caption suggests the presence
                                  of problematic image properties (e.g., 'arrows') likely to corrupt
                                  a dataset intended for machine learning. Defaults to ``False``.
        :type use_image_caption: ``bool``
        :return: a DataFrame with the record information.
                 If ``image_size`` is not None, images will also be harvested and cached.
        :rtype: ``Pandas DataFrame``
        :raises ``ValueError``: if ``search()`` has not been called.
        """
        if self.current_query is None:
            raise ValueError("`search()` must be called before `pull()`.")

        # Note the time the pull request was made
        self._pull_time = datetime.now()

        # Pull Records
        if new_records_pull:
            self.records_db = self._Records.records_pull(search_url=self.current_search_url,
                                                         to_harvest=self._current_search_to_harvest,
                                                         total=self.current_search_total,
                                                         query=self.current_query,
                                                         pull_time=self._pull_time,
                                                         records_sleep_time=records_sleep_time,
                                                         clinical_cases_only=clinical_cases_only,
                                                         download_limit=download_limit)

        # Pull Images
        if isinstance(image_size, str):
            # Pull the images.
            self.records_db = self._Images.pull_images(records_db=self.records_db,
                                                       image_size=image_size,
                                                       pull_time=self._pull_time.strftime(TIME_FORMAT),
                                                       images_sleep_time=images_sleep_time,
                                                       use_image_caption=use_image_caption)

            # Add the new records_db datafame with the existing `cache_records_db`.
            self._openi_cache_records_db_handler(current_records_db=self.cache_records_db,
                                                 records_db_update=self.records_db)

            # Delete the 'databases/__temp__' folder.
            shutil.rmtree(self._Images.temp_directory_path, ignore_errors=True)

        return self.records_db



















