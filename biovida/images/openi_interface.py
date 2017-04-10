# coding: utf-8

"""

    Open-i Interface
    ~~~~~~~~~~~~~~~~

"""
import os
import shutil
import pickle
import requests
import numpy as np
import pandas as pd
from math import floor
from copy import deepcopy
from datetime import datetime

from biovida import __version_numeric__

# General Image Support Tools
from biovida.images._image_tools import NoResultsFound, sleep_with_noise

# Database Management
from biovida.images.image_cache_mgmt import (_records_db_merge,
                                             _openi_image_relation_map,
                                             _record_update_dbs_joiner,
                                             _prune_rows_with_deleted_images)

# Interface Support tools
from biovida.images._interface_support.shared import save_records_db

from biovida.images._interface_support.openi.openi_support_tools import (iter_join,
                                                                         url_combine,
                                                                         null_convert,
                                                                         ImageProblemBasedOnText,
                                                                         nonessential_openi_columns)

# Open-i API Parameters Information
from biovida.images._interface_support.openi.openi_parameters import openi_search_information
from biovida.images._interface_support.openi._openi_image_id_processing import image_id_short_gen
from biovida.images._interface_support.openi.openi_text_processing import openi_raw_extract_and_clean

# Cache Management
from biovida.support_tools._cache_management import package_cache_creator

# General Support Tools
from biovida.support_tools.support_tools import (tqdm,
                                                 cln,
                                                 camel_to_snake_case,
                                                 data_frame_col_drop,
                                                 list_to_bulletpoints)


# ----------------------------------------------------------------------------------------------------------
# Searching
# ----------------------------------------------------------------------------------------------------------


class _OpeniSearch(object):
    """

    Searching the Open-i Database.

    """

    def __init__(self):
        self._root_url = 'https://openi.nlm.nih.gov'
        self.current_search = None
        self.search_dict, self.ordered_params = openi_search_information()
        self.search_params = ('image_type', 'rankby', 'article_type', 'subset', 'collection', 'fields',
                              'specialties', 'video', 'exclusions')

    @staticmethod
    def _openi_search_special_case(search_param, blocked, passed):
        """

        Check for invalid search requests when contradictions are possible,
        e.g., `passed` cannot contain `True` and `False`. 

        :param search_param: currently, only `rankby` is valid.
        :param blocked: mutually exclusive (i.e., all these items cannot be passed together).
        :param passed: values actually passed to `search_param`.
        """
        if all(b in passed for b in blocked):
            raise ValueError("`{0}` can only contain one of:\n{1}".format(
                search_param, list_to_bulletpoints(blocked)))

    def _openi_search_check(self, search_arguments):
        """

        Method to check for invalid search requests.

        :param search_arguments: arguments to pass to Open-i's REST API.
        :type search_arguments: ``dict``
        """
        general_error_msg = "'{0}' is not valid for `{1}`.\nValid values for `{1}`:\n{2}"

        if not isinstance(search_arguments['query'], str):
            raise ValueError("`query` must be a string.")

        # Check all other params
        for k, v in search_arguments.items():
            # Halt if query or `v` is NoneType
            if k != 'query' and v is not None:
                # Check type
                if not isinstance(v, (list, tuple)) and v is not None:
                    raise ValueError("Only `lists`, `tuples` or `None` "
                                     "may be passed to `{0}`.".format(k))
                for i in v:
                    if not isinstance(i, str):
                        raise ValueError("`tuples` or `lists` passed to `{0}` "
                                         "must only contain strings.".format(k))
                    # Check if `i` can be converted to a param understood by the Open-i API
                    if i not in self.search_dict[k][1].keys():
                        raise ValueError(general_error_msg.format(
                            i, k, list_to_bulletpoints(self.search_dict[k][1].keys())))

                if k == 'rankby':
                    self._openi_search_special_case(k, blocked=['newest', 'oldest'], passed=v)

    @staticmethod
    def _exclusions_image_type_merge(args, exclusions):
        """
        
        Merge Image type with Exclusions.
        This must be performed because while this module
        separates 'exclusions' it is actually under the 
        'image_type' ('&it') param. in the open-i REST API.
        
        :param args: all arguments passed to ``search()`` to pass to Open-i's REST API.
        :type args: ``dict``
        :param exclusions: a list of exclusions to pass to '&it'.
        :type exclusions: ``list`` or ``None``
        """
        if not isinstance(exclusions, (list, tuple)) and exclusions is not None:
            raise ValueError('`exclusions` must be a `list`, `tuple` or `None`.')

        if exclusions is None or (isinstance(exclusions, (list, tuple)) and not len(exclusions)):
            return args

        if any(e not in ['graphics', 'multipanel'] for e in exclusions):
            raise ValueError("`exclusions` must only include one or all of: 'graphics', 'multipanel'.")

        # Handle handle tuples, then `None`s (with the 'else []').
        args['image_type'] = list(args['image_type']) if isinstance(args['image_type'], (list, tuple)) else []

        # Merge `exclusions` with `image_type`
        args['image_type'] += list(map(lambda x: 'exclude_{0}'.format(x), exclusions))

        return args

    def _search_url_formatter(self, api_search_transform):
        """

        Format ``api_search_transform`` as a complete URL to pass to Open-i.

        :param api_search_transform: a dictionary of parameters and keys in the precise form Open-i expects.
        :type api_search_transform: ``dict``
        :return: a formatted `GET` request.
        :rtype: ``str``
        """
        if 'query' in api_search_transform:
            api_search_transform['query'] = cln(api_search_transform['query']).replace(' ', '+')
        else:
            raise ValueError("No `query` detected.")

        search_term = ""
        for p in self.ordered_params:
            if p in api_search_transform:
                search_term += "{0}{1}={2}".format(("?" if p == 'query' else ""), p, api_search_transform[p])

        return "{0}/retrieve.php{1}".format(self._root_url, search_term)

    @staticmethod
    def _search_probe(search_query, print_results):
        """

        Request results from Open-i.

        :param search_query: a fully format Open-i REST GET command.
        :type search_query: ``str``
        :param print_results: print the number of results found
        :type print_results: ``bool``
        :return: a tuple of the form ``(total number of results, the first result returned)``
        :rtype: ``tuple``
        """
        # Get a sample request
        sample = requests.get(search_query + "&m=1&n=1").json()

        try:
            total = int(float(sample['total']))
        except:
            raise ValueError("Could not obtain total number of results from the Open-i API.")

        if total < 1:
            raise NoResultsFound("\nPlease Try Refining Your Search.")

        if print_results:
            print("\nResults Found: %s." % ('{:,.0f}'.format(total)))

        return total, sample['list'][0]

    def options(self, search_parameter=None, print_options=True):
        """

        Options for parameters of `openi_search()`.

        :param search_parameter: one of: 'image_type', 'rankby', 'article_type', 'subset', 'collection', 'fields',
                                         'specialties', 'video' or `exclusions`. If ``None``, print the parameters
                                         of ``search()``.
        :param print_options: if True, pretty print the options, else return as a ``list``.
        :return: a list of valid values for a given search `search_parameter`.
        :rtype: ``list``
        """
        if search_parameter is not None and not isinstance(search_parameter, str):
            raise TypeError("`search_parameter` must be a string or None.")

        # Terms to blocked from displaying to users if search_parameter != 'exclusions'
        exclusions = ['exclude_graphics', 'exclude_multipanel']

        if search_parameter is None:
            opts = self.search_params
        elif search_parameter == 'exclusions':
            opts = [i.split("_")[1] for i in exclusions]
        else:
            # Get the relevant dict of params
            search_dict_against_param = self.search_dict.get(cln(search_parameter).strip().lower(), None)

            if search_dict_against_param is None:
                raise ValueError("'{0}' is not a valid parameter to pass to the Open-i API.".format(search_parameter))

            # Remove exclusions term
            opts = [i for i in search_dict_against_param[1].keys() if i not in exclusions]
            if not len(opts):
                raise ValueError("Relevant options for '{0}'.".format(search_parameter))

        if print_options:
            print(list_to_bulletpoints(opts))
        else:
            return opts

    @staticmethod
    def _search_clean(k, v):
        """

        Clean the search terms (strings) passed to `OpeniInterface().search()`.

        :param k: Open-i param.
        :type k: ``str``
        :param v: value passed to ``k`` .
        :type v: ``list``
        :return: ``v`` with the strings it contained cleaned. 
        :rtype: ``list``
        """
        return [cln(i).replace(' ', '_').lower() for i in v] if k != 'query' and v is not None else v

    def _api_url_terms(self, k, v):
        """

        Convert values passed into a valid REST GET string
        from a dictionary.

        :param k: Open-i param.
        :type k: ``str``
        :param v: value passed to ``k`` .
        :type v: ``list``
        :return: see description.
        :rtype: ``str``
        """
        return ','.join([self.search_dict[k][1][i] for i in v]) if k != 'query' else v

    def _api_url_param(self, k):
        """
        
        Convert param names into a form the API will understand.
        E.g., 'video' --> '&vid'.
        
        :param k: a human readable term into an actual Open-i param.
        :return: see description.
        :rtype: ``str``
        """
        return self.search_dict[k][0] if k != 'query' else 'query'

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
        args_cleaned = {k: v for k, v in deepcopy(locals()).items()
                        if k not in ['self', 'print_results']}

        # Extract the function arguments and format values.
        args = {k: [v] if isinstance(v, str) and k != 'query' else v
                for k, v in args_cleaned.items() if k != 'exclusions'}

        # Merge `image_type` with `exclusions`
        args = self._exclusions_image_type_merge(args, exclusions)

        # Get the arguments
        search_arguments = {k: '' if k == 'query' and v is None else self._search_clean(k, v)
                            for k, v in args.items()}

        # Save search query
        self.current_search = {k: v for k, v in deepcopy(search_arguments).items() if v is not None}

        # Check if search terms are valid
        self._openi_search_check(search_arguments=search_arguments)

        # Perform transformation
        api_search_transform = {self._api_url_param(k): self._api_url_terms(k, v)
                                for k, v in search_arguments.items() if v is not None}

        # Format `api_search_transform`
        search_url = self._search_url_formatter(api_search_transform=api_search_transform)

        # Unpack the probe containing information on the total number of results and list of results to harvest
        current_search_total, current_search_to_harvest = self._search_probe(search_url, print_results)

        return {"query": args_cleaned,
                "search_url": search_url,
                "current_search_total": current_search_total,
                "current_search_to_harvest": current_search_to_harvest}


# ----------------------------------------------------------------------------------------------------------
# Pull Records from the NIH's Open-i API
# ----------------------------------------------------------------------------------------------------------


class _OpeniRecords(object):
    """

    Obtaining Records from the Open-i Database.

    :param root_url: suggested: 'https://openi.nlm.nih.gov'
    :type root_url: ``str``
    :param date_format: suggested: "%d/%m/%Y" (consider leaving as datetime)
    :type date_format: ``str``
    :param verbose: if ``True`` print additional details.
    :type verbose: ``bool``
    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                   one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str``
    :param req_limit: Defaults to 30 (max allowed by Open-i; see: https://openi.nlm.nih.gov/services.php?it=xg).
    :type req_limit: ``int``
    """

    def __init__(self, root_url, date_format, verbose, cache_path, req_limit=30):
        self.root_url = root_url
        self.date_format = date_format
        self._verbose = verbose
        self._cache_path = cache_path
        self.req_limit = req_limit

        self.records_db = None
        self.download_limit = 100  # set to reasonable default.

        # Sleep Time
        self.records_sleep_time = None

    def openi_bounds(self, total):
        """
        
        Generate a list of tuples which define
        the upper and lower bounds of each request to Open-i
        for records, e.g., ``[(1, 30), ..., (90, total)]``.

        :param total: the total number of results for a given search.
        :type total: ``int``
        :return: see description.
        :rtype: ``list``
        """
        end = 1
        bounds = list()

        if total < 1:
            raise ValueError("'{0}' is an invalid value for total.".format(str(total)))

        if self.download_limit is not None and not isinstance(self.download_limit, int):
            raise ValueError("`download_limit` must be an `int` or `None`.")
        if isinstance(self.download_limit, int) and self.download_limit < 1:
            raise ValueError("`download_limit` cannot be less than 1.")

        if total < self.req_limit:
            return [(1, total)]
        elif self.download_limit is not None and total > self.download_limit:
            download_no = self.download_limit
        else:
            download_no = total

        # Compute the number of steps and floor
        n_steps = int(floor(download_no / self.req_limit))

        for _ in range(n_steps):
            bounds.append((end, end + (self.req_limit - 1)))
            end += self.req_limit

        # Compute the remainder
        remainder = download_no % self.req_limit

        # Add remaining part, if nonzero
        if remainder != 0:
            bounds += [(download_no - remainder + 1, download_no)]

        return bounds, download_no

    @staticmethod
    def openi_bounds_formatter(bounds):
        """

        Format the computed bounds for the Open-i API.

        :param bounds: as returned by ``_OpeniPull().openi_bounds()``
        :type bounds: ``list``
        :return: the inner tuples in ``bounds`` converted to strings for Open-i.
        :rtype: ``list``
        """
        return ["&m={0}&n={1}".format(i[0], i[1]) for i in bounds]

    def date_formatter(self, date_dict):
        """

        Format dates in the 'journal_date' column.

        :param date_dict: a dictionary of the date information
        :type date_dict: ``dict``
        :return: ``date_dict`` as a datetime object.
        :rtype: ``datetime``
        """
        # ToDo: return a datetime object instead of a string.
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

        # ToDo: add date format guessing -- e.g., ``pd.to_datetime(..., infer_datetime_format=True)``.
        try:
            return datetime(cleaned_info[0], cleaned_info[1], cleaned_info[2]).strftime(self.date_format)
        except:
            return None

    @staticmethod
    def harvest_vector(request_rslt):
        """

        Defines the terms to harvest from the results returned by the Open-i API.
        
        Note:
        
            This method assumes a maximum of one nest in ``request_rslt`` (i.e., ``key: {key: value...}``).

        :param request_rslt: request 'records' from Open-i.
        :type request_rslt: ``dict``
        :return: Convert ``request_rslt`` into a list. Nested dictionaries in ``request_rslt`` are added 
                 the list this method returns as tuples.
        :rtype: ``list``
        """
        time_units = ['day', 'month', 'year']

        to_harvest = list()
        for k, v in request_rslt.items():
            if isinstance(v, str):
                to_harvest.append(k)
            elif isinstance(v, dict) and any(dmy in map(lambda x: x.lower(), v.keys()) for dmy in time_units):
                to_harvest.append(k)
            elif isinstance(v, dict):
                for i in v.keys():
                    to_harvest.append((k, i))

        return to_harvest

    def openi_block_harvest(self, url, bound, to_harvest):
        """

        Download a single block (or 'chunk') of records, e.g, 'records' 1-30.

        :param url: the GET URL for the current search.
        :type url: ``str``
        :param bound: the bound to append to the GET URL, e.g., "&m=1&n=30".
        :type bound: ``str``
        :param to_harvest: yield of ``harvest_vector()``.
        :type to_harvest: ``list``
        :return: a list of dictionaries, where the keys of the dictionaries will become
                 the columns of the ``records_db`` dataframe.
        :rtype: ``list``
        """
        # Request data from the Open-i servers
        req = requests.get(url + bound).json()['list']

        root_url_columns = ('detailed_query_url', 'get_article_figures',
                            'similar_in_collection', 'similar_in_results')

        def append_root_url(item):
            """Check whether or not to add `self.root_url` to a column."""
            return 'img_' in item or any(c == item for c in root_url_columns)

        list_of_dicts = list()
        for item in req:
            # Create an item_dict the dict
            item_dict = dict()
            # Populate current `item_dict`
            for j in to_harvest:
                if isinstance(j, (list, tuple)):
                    item_dict[iter_join(j)] = null_convert(item.get(j[0], {}).get(j[1], None))
                elif j == 'journal_date':
                    item_dict[j] = null_convert(self.date_formatter(item.get(j, None)))
                elif append_root_url(camel_to_snake_case(j)):
                    item_dict[j] = url_combine(self.root_url, null_convert(item.get(j, None)))
                else:
                    item_dict[j] = null_convert(item.get(j, None))

            list_of_dicts.append(item_dict)

        return list_of_dicts

    def _records_pull_engine(self, bounds_list, search_url, to_harvest, records_sleep_time, download_no):
        """
        
        Download all records requested by the user.

        :param bounds_list: a list of bounds (chunks), e.g., ``["&m=1&n=30", "&m=31&n=59"...]``. 
        :type bounds_list: ``list``
        :param search_url: search URL as generated by ``_OpeniSearch().search()``.
        :type search_url: ``str``
        :param to_harvest: yield of ``harvest_vector()``.
        :type to_harvest: ``list``
        :param records_sleep_time: (every x downloads, period of time [seconds])
        :type records_sleep_time: ``tuple`` or ``None``
        :param download_no: the number of records to download.
        :type download_no: ``int``
        :return: a list of dictionaries where the keys are future columns of the ``records_db`` dataframe.
        :rtype: ``list``
        """
        if self._verbose:
            print("\nNumber of Records to Download: {0} (chunk size: {1} records).".format(
                '{:,.0f}'.format(download_no), str(self.req_limit)))

        do_sleep = isinstance(records_sleep_time, (list, tuple)) and len(records_sleep_time) == 2

        harvested_data = list()
        for c, bound in enumerate(tqdm(bounds_list, desc='Obtaining Records', disable=not self._verbose), start=1):
            if do_sleep and c % records_sleep_time[0] == 0:
                sleep_with_noise(amount_of_time=records_sleep_time[1])
            harvested_data += self.openi_block_harvest(search_url, bound=bound, to_harvest=to_harvest)

        return harvested_data

    def records_pull(self,
                     search_url,
                     to_harvest,
                     total,
                     query,
                     pull_time,
                     records_sleep_time,
                     clinical_cases_only,
                     download_limit=None,
                     **kwargs):
        """

        'Walk' along the search query and harvest the data
        in the 'chunk size' given by ``req_limit``.

        :param search_url: search URL as generated by ``_OpeniSearch().search()``.
        :type search_url: ``str``
        :param to_harvest: a list of results to harvest as generated by ``_OpeniSearch().search()``
                          (AKA 'current_search_to_harvest').
        :type to_harvest: ``list``
        :param total: the total number of results (current_search_total) as generated by ``_OpeniSearch().search()``.
        :type total: ``int``
        :param query: a dictionary of parameters and values passed to ``_OpeniSearch().search()``.
        :type query: ``dict``
        :param pull_time: yield of ``datetime.now()`` as as evolved inside ``OpeniInterface.pull()``.
        :type pull_time: ``datetime.datetime``
        :param records_sleep_time: (every x downloads, period of time [seconds])
        :type records_sleep_time: ``tuple`` or ``None``
        :param clinical_cases_only: see ``OpeniInterface().pull()``
        :type clinical_cases_only: ``bool``
        :param download_limit: see ``OpeniInterface().pull()``
        :type download_limit: ``None`` or ``int``
        :return: a complete ``records_db``
        :rtype: ``Pandas DataFrame``
        """
        if isinstance(download_limit, int):
            self.download_limit = download_limit
        elif download_limit is None:
            self.download_limit = None
        else:
            raise TypeError("`download_limit` must be an `int` or `None`")

        # Get a list of lists with the bounds
        bounds, download_no = self.openi_bounds(total)

        # Compute a list of search ranges to pass to the Open-i API
        bounds_list = self.openi_bounds_formatter(bounds)

        harvest = self._records_pull_engine(bounds_list=bounds_list,
                                            search_url=search_url,
                                            to_harvest=self.harvest_vector(to_harvest),
                                            records_sleep_time=records_sleep_time,
                                            download_no=download_no)

        records_db = pd.DataFrame(harvest).fillna(np.NaN)

        if kwargs.get('return_raw', False):
            return records_db

        records_db = openi_raw_extract_and_clean(data_frame=records_db,
                                                 clinical_cases_only=clinical_cases_only,
                                                 verbose=self._verbose,
                                                 cache_path=self._cache_path)

        records_db['query'] = [query] * records_db.shape[0]
        records_db['pull_time'] = [pull_time] * records_db.shape[0]
        records_db['biovida_version'] = [__version_numeric__] * records_db.shape[0]

        self.records_db = records_db
        return self.records_db


# ----------------------------------------------------------------------------------------------------------
# Image Harvesting
# ----------------------------------------------------------------------------------------------------------


class _OpeniImages(object):
    """
    
    Obtaining Images from the Open-i Database.

    :param image_save_location: directory to save the images to.
    :type image_save_location: ``str``
    :param database_save_location: system path to 'openi/databases'.
    :type database_save_location: ``str``
    :param verbose: if ``True`` print additional details.
    :type verbose: ``bool``
    """

    def __init__(self, image_save_location, database_save_location, verbose):
        self.image_save_location = image_save_location
        self._verbose = verbose

        # Database
        self.records_db_images = None
        self.real_time_update_db = None

        self.temp_directory_path = os.path.join(database_save_location, "__temp__")

    def _create_temp_directory_path(self):
        """

        Check if ``self.temp_directory_path`` exists. If not, create it.

        """
        if not os.path.isdir(self.temp_directory_path):
            os.makedirs(self.temp_directory_path)

    def _instantiate_real_time_update_db(self, db_index):
        """

        Create the ``real_time_update_db`` and define the path to the location where it will be saved.

        :param db_index: the index of the ``real_time_update_db`` dataframe (should be from ``records_db``).
        :type db_index: ``Pandas Series``
        """
        real_time_update_columns = ['cached_images_path', 'download_success']

        # Instantiate
        db = pd.DataFrame(columns=real_time_update_columns, index=db_index).replace({np.NaN: None})
        self.real_time_update_db = db

    def _title_image(self, url, image_size):
        """

        Generate a title for the images.

        :param url: the URL to the image.
        :type url: ``str``
        :param image_size: the size of the image, e.g., 'large'.
        :type image_size: ``str``
        :return: see description. 
        :type: ``str``
        """
        # Get the actual file name
        base = os.path.basename(url)

        # Separate the name from the image type
        b_name, image_format = os.path.splitext(base)

        # Generate and clean strings to populate the name format below. Note: 1 = file number
        # (in case medpix has images with multiple segments -- though, it doesn't appear to currently.)
        replacement_terms = map(lambda x: cln(x), (str(1), b_name, image_size, image_format.replace(".", "")))

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

        def proceed_with_download(image_save_path):
            return not os.path.isfile(image_save_path)

        try:
            # Only download if the file does not already exist in the cache.
            if proceed_with_download(image_save_path):
                if block:
                    raise ImageProblemBasedOnText

                # Get the image
                page = requests.get(image_url)
                # Save to disk
                with open(image_save_path, 'wb') as image:
                    image.write(page.content)
                image_downloaded = 1

            self.real_time_update_db.set_value(index, 'cached_images_path', image_save_path)
            self.real_time_update_db.set_value(index, 'download_success', True)
        except:
            self.real_time_update_db.set_value(index, 'cached_images_path', np.NaN)
            self.real_time_update_db.set_value(index, 'download_success', False)

        return image_downloaded

    def _pull_images_engine(self, harvesting_information, images_sleep_time, image_size, use_image_caption):
        """

        Use ``_individual_image_harvest()`` to download all of the data (images) in ``harvesting_information``.

        :param harvesting_information: as evolved inside ``harvesting_information``
        :type harvesting_information: ``list``
        :param images_sleep_time: see ``pull_images()``
        :type images_sleep_time: ``tuple`` or ``None``
        :param image_size: see ``pull_images()``
        :type image_size: ``str``
        :param use_image_caption: if ``True`` block downloading of an image if its caption suggests the presence
                                  of problematic image properties (e.g., 'arrows') likely to corrupt
                                  a dataset intended for machine learning. Defaults to ``False``.
        :type use_image_caption: ``bool``
        """
        def block_decision(ipt):
            """Decide whether or not to block the downloading."""
            return use_image_caption == True and isinstance(ipt, (list, tuple)) and len(ipt)

        do_sleep = isinstance(images_sleep_time, (list, tuple)) and len(images_sleep_time) == 2
            
        download_count = 0
        for index, image_url, image_problems_text in tqdm(harvesting_information, desc='Obtaining Images',
                                                          disable=not self._verbose):
            # Generate the save path for the image
            image_save_path = self._title_image(url=image_url, image_size=image_size)

            # Save the image
            download_count += self._individual_image_harvest(index=index,
                                                             image_url=image_url,
                                                             image_save_path=image_save_path,
                                                             block=block_decision(image_problems_text))

            # Sleep when `download_count` 'builds up' to images_sleep_time[0].
            if do_sleep and download_count == images_sleep_time[0]:
                sleep_with_noise(amount_of_time=images_sleep_time[1])
                download_count = 0  # reset

    def pull_images(self,
                    records_db,
                    image_size,
                    images_sleep_time,
                    use_image_caption):
        """

        Pull images based in ``records_db``.

        :param records_db: yield of ``_OpeniRecords().records_pull()``
        :type records_db: ``Pandas DataFrame``
        :param image_size: one of 'grid150', 'large', 'thumb' or 'thumb_large'.
        :type image_size: ``str``
        :param images_sleep_time: tuple of the form: ``(every x downloads, period of time [seconds])``. Defaults to ``(10, 1.5)``.
                                   Note: noise is randomly added to the sleep time by sampling from a normal distribution
                                   (with mean = 0, sd = 0.75).
        :type images_sleep_time: ``tuple`` or ``None``
        :param use_image_caption: if ``True`` block downloading of an image if its caption suggests the presence
                                  of problematic image properties (e.g., 'arrows') likely to corrupt
                                  a dataset intended for machine learning. Defaults to ``False``.
        :type use_image_caption: ``bool``
        :return: `records_db` with the addition of `cached_images_path` and `download_success` columns.
        :rtype: ``Pandas DataFrame``
        """
        self._create_temp_directory_path()
        self.records_db_images = records_db.copy(deep=True)

        settings_path = os.path.join(self.temp_directory_path, "image_pull_settings.p")
        pickle.dump({k: v for k, v in locals().items() if k not in ('self', 'settings_path')},
                    open(settings_path, "wb"))

        self._instantiate_real_time_update_db(db_index=self.records_db_images.index)

        if image_size not in ('grid150', 'large', 'thumb', 'thumb_large'):
            raise ValueError("`image_size` must be one of: 'grid150', 'large', 'thumb' or 'thumb_large'.")
        image_column = "img_{0}".format(image_size)

        # Extract needed information from the `records_db_images` dataframe to loop over.
        harvesting_information = list(zip(*[self.records_db_images.index,
                                            self.records_db_images[image_column],
                                            self.records_db_images['image_problems_from_text']]))

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
    :param verbose: if ``True`` print additional details.
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
            temp_dir = self._Images.temp_directory_path
            latent_pickles = [os.path.join(temp_dir, i) for i in os.listdir(temp_dir) if i.endswith(".p")]
            if len(latent_pickles):
                with open(latent_pickles[0], "rb") as f:
                    settings_dict = pickle.load(f)
                settings_dict_for_pull = {k: v for k, v in settings_dict.items() if k not in ['records_db']}
                settings_dict_for_pull['new_records_pull'] = False  # adding this separately in `pull` breaks in python2

                print("\nResuming Download...")
                self.load_records_db(records_db=settings_dict['records_db'])
                self.pull(**settings_dict_for_pull)

            # `temp_directory_path` will be destroyed when `pull()` exits successfully

    def _openi_cache_records_db_handler(self):
        """

        1. if cache_records_db.p doesn't exist, simply save ``records_db_update`` to disk.
        2. if cache_records_db.p does exist, merge with ``records_db_update`` and then save to disk.

        """
        def rows_to_conserve_func(x):
            return x['download_success'] == True

        def openi_duplicates_handler(data_frame):
            duplicates_subset_columns = ['img_grid150', 'img_large', 'img_thumb',
                                         'img_thumb_large', 'query', 'download_success']

            # Note: Typically biovida_version' and 'pull_time' will be 'in sync'.
            # However, if they are not, this function gives preference to 'biovida_version'
            # s.t. the data harvested with the latest version is given priority.
            data_frame = data_frame.sort_values(by=['biovida_version', 'pull_time'])
            return data_frame.drop_duplicates(subset=duplicates_subset_columns, keep='last')

        if self.cache_records_db is None and self.records_db is None:
            raise ValueError("`current_records_db` and `records_db` cannot both be None.")
        elif self.cache_records_db is not None and self.records_db is None:
            data_frame = self.cache_records_db
            self.cache_records_db = data_frame[
                data_frame.apply(rows_to_conserve_func, axis=1)].reset_index(drop=True)
        elif self.cache_records_db is None and self.records_db is not None:
            data_frame = _openi_image_relation_map(self.records_db)
            self.cache_records_db = data_frame[
                data_frame.apply(rows_to_conserve_func, axis=1)].reset_index(drop=True)
        else:
            self.cache_records_db = _records_db_merge(interface_name='OpeniInterface',
                                                      current_records_db=self.cache_records_db,
                                                      records_db_update=self.records_db,
                                                      columns_with_dicts=('query', 'parsed_abstract'),
                                                      duplicates=openi_duplicates_handler,
                                                      rows_to_conserve_func=rows_to_conserve_func,
                                                      pre_return_func=image_id_short_gen)

        # Save to disk
        self._save_cache_records_db()

        # Destroy the '__temp__' folder
        shutil.rmtree(self._Images.temp_directory_path, ignore_errors=True)

    def __init__(self, cache_path=None, verbose=True):
        self._cache_path = cache_path
        self._verbose = verbose
        self._root_url = 'https://openi.nlm.nih.gov'
        self._date_format = '%d/%m/%Y'

        # Generate Required Caches
        _, self._created_image_dirs = package_cache_creator(sub_dir='images',
                                                            cache_path=cache_path,
                                                            to_create=['openi'],
                                                            nest=[('openi', 'aux'), ('openi', 'raw'),
                                                                  ('openi', 'databases')],
                                                            verbose=verbose,
                                                            requires_medpix_logo=True)

        self._ROOT_PATH = self._created_image_dirs['ROOT_PATH']

        # Instantiate Classes
        self._Search = _OpeniSearch()

        self._Records = _OpeniRecords(root_url=self._root_url,
                                      date_format=self._date_format,
                                      verbose=verbose,
                                      cache_path=cache_path)

        self._Images = _OpeniImages(image_save_location=self._created_image_dirs['raw'],
                                    database_save_location=self._created_image_dirs['databases'],
                                    verbose=verbose)

        self._pull_time = None
        self.current_query = None
        self.current_search_url = None
        self.current_search_total = None
        self._current_search_to_harvest = None

        self.records_db = None

        # Path to cache record db
        self._cache_records_db_save_path = os.path.join(self._created_image_dirs['databases'],
                                                        'openi_cache_records_db.p')

        # Load the cache record database, if it exists
        if os.path.isfile(self._cache_records_db_save_path):
            self._load_prune_cache_records_db(load=True)
        else:
            self.cache_records_db = None

        # Load in a latent database in 'databases/__temp__', if one exists
        self._latent_temp_dir()

    def save_records_db(self, path):
        """

        Save the current ``records_db``.

        :param path: a system path ending with the '.p' file extension.
        :type path: ``str``
        """
        save_records_db(data_frame=self.records_db, path=path)

    def load_records_db(self, records_db):
        """

        Load a ``records_db``

        :param records_db: a system path or ``records_db`` itself.
        :type records_db: ``str`` or ``Pandas DataFrame``
        """
        if isinstance(records_db, pd.DataFrame):
            self.records_db = records_db
        else:
            self.records_db = pd.read_pickle(records_db)
        self._pull_time = self.records_db['pull_time'].iloc[0]
        last_query = self.records_db['query'].iloc[0]
        last_query['print_results'] = False
        self.search(**last_query)

    @property
    def records_db_short(self):
        """Return `records_db` with nonessential columns removed."""
        return data_frame_col_drop(self.records_db, nonessential_openi_columns, 'records_db')

    @property
    def cache_records_db_short(self):
        """Return `cache_records_db` with nonessential columns removed."""
        return data_frame_col_drop(self.cache_records_db, nonessential_openi_columns, 'cache_records_db')

    def options(self, search_parameter=None, print_options=True):
        """

        Options for parameters of ``search()``.

        :param search_parameter: one of: 'image_type', 'rankby', 'article_type', 'subset', 'collection', 'fields',
                                         'specialties', 'video' or `exclusions`. If ``None``, print the parameters
                                         of ``search()``.
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

        self.current_query = search['query']
        self.current_search_url = search['search_url']
        self.current_search_total = search['current_search_total']
        self._current_search_to_harvest = search['current_search_to_harvest']

    def pull(self,
             image_size='large',
             records_sleep_time=(10, 1.5),
             images_sleep_time=(10, 1.5),
             download_limit=100,
             clinical_cases_only=False,
             use_image_caption=False,
             new_records_pull=True):
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

        :param image_size: one of: 'large', 'grid150', 'thumb', 'thumb_large' or ``None``. Defaults to 'large'.
                          If ``None``, no attempt will be made to download images.

                        .. warning::

                                The analyses performed by the ``image_processing.OpeniImageProcessing`` class are
                                most accurate with large images.

        :type image_size: ``str`` or ``None``
        :param records_sleep_time: tuple of the form: ``(every x downloads, period of time [seconds])``. Defaults to ``(10, 1.5)``.
                                   Note: noise is randomly added to the sleep time by sampling from a normal distribution
                                   (with mean = 0, sd = 0.75).
        :type records_sleep_time: ``tuple`` or ``None``
        :param images_sleep_time: tuple of the form: ``(every x downloads, period of time [seconds])``. Defaults to ``(10, 1.5)``.
                                  Note: noise is randomly added to the sleep time by sampling from a normal distribution
                                  (with mean = 0, sd = 0.75).
        :type images_sleep_time: ``tuple`` or ``None``
        :param download_limit: max. number of results to download. If ``None``, no limit will be imposed
                              (not recommended). Defaults to 100.
        :type download_limit: ``None`` or ``int``
        :param clinical_cases_only: if ``True`` require that the data harvested is of a clinical case. Specifically,
                                    this parameter requires that 'article_type' is one of: 'encounter', 'case_report'.
                                    Defaults to ``False``.

                    .. note::

                        If ``True``, this parameter will often result in fewer records being returned than
                        the ``download_limit``.

        :type clinical_cases_only: ``bool``
        :param use_image_caption: if ``True`` block downloading of an image if its caption suggests the presence
                                  of problematic image properties (e.g., 'arrows') likely to corrupt a dataset.
                                  Defaults to ``False``.
        :type use_image_caption: ``bool``
        :param new_records_pull: if ``True``, download the data for the current search. If ``False``, use ``INSTANCE.records_db``.

            .. note::

               Setting ``new_records_pull=False`` can be useful if one wishes to initially set ``image_size=None``,
               truncate or otherwise modify ``INSTANCE.records_db`` and then download images.

        :type new_records_pull: ``bool``
        :return: a DataFrame with the record information.
                 If ``image_size`` is not None, images will also be harvested and cached.
        :rtype: ``Pandas DataFrame``
        :raises ``ValueError``: if ``search()`` has not been called.
        """
        if not new_records_pull and not isinstance(self.records_db, pd.DataFrame):
            raise TypeError("`records_db` is not a dataframe.")

        if new_records_pull:
            if not isinstance(self.current_search_url, str):
                raise ValueError("`search()` must be called before `pull()`.")
            self._pull_time = datetime.now()
            self.records_db = self._Records.records_pull(search_url=self.current_search_url,
                                                         to_harvest=self._current_search_to_harvest,
                                                         total=self.current_search_total,
                                                         query=self.current_query,
                                                         pull_time=self._pull_time,
                                                         records_sleep_time=records_sleep_time,
                                                         clinical_cases_only=clinical_cases_only,
                                                         download_limit=download_limit)
        elif not isinstance(self.records_db, pd.DataFrame):
            raise TypeError("`records_db` is not a DataFrame.")

        if isinstance(image_size, str):
            self.records_db = self._Images.pull_images(records_db=self.records_db,
                                                       image_size=image_size,
                                                       images_sleep_time=images_sleep_time,
                                                       use_image_caption=use_image_caption)

            # Add the new records_db datafame with the existing `cache_records_db`.
            self._openi_cache_records_db_handler()

        return self.records_db
