"""

    Open-i Interface
    ~~~~~~~~~~~~~~~~

"""
# Imports
import re
import os
import pickle
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import floor
from time import sleep
from copy import deepcopy
from warnings import warn
from datetime import datetime
from collections import Counter

# Image Support tools
from biovida.images._openi_support_tools import iter_join
from biovida.images._openi_support_tools import null_convert
from biovida.images._openi_support_tools import numb_extract
from biovida.images._openi_support_tools import url_combine

# Open-i API Parameters Information
from biovida.images._resources.openi_parameters import openi_image_type_params
from biovida.images._resources.openi_parameters import openi_search_information
from biovida.images._resources.openi_parameters import openi_article_type_params

# Tools for Text Feature Extraction
from biovida.images.text_processing import feature_extract

# Cache Managment
from biovida.support_tools._cache_management import package_cache_creator

# General Support Tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import same_dict
from biovida.support_tools.support_tools import unique_dics
from biovida.support_tools.support_tools import hashable_cols
from biovida.support_tools.support_tools import camel_to_snake_case
from biovida.support_tools.support_tools import list_to_bulletpoints

# To install scipy: brew install gcc; pip3 install Pillow

from biovida.support_tools.printing import pandas_pprint

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

        self.current_search_url = None
        self.current_search_total = None
        self._current_search_to_harvest = None

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

    def search(self
               , query
               , image_type=None
               , rankby=None
               , article_type=None
               , subset=None
               , collection=None
               , fields=None
               , specialties=None
               , video=None
               , exclusions=None
               , print_results=True):
        """

        Tool to generate a search term (URL) for the NIH's Open-i API.
        The computed term is stored as a class attribute (``INSTANCE.current_search_url``)

        :param query: a search term. ``None`` will be converted to an empty string.
        :type query: ``str`` or ``None``
        :param image_type: see ``OpeniInterface().options('image_type')`` for valid values.
        :type image_type: ``list``, ``tuple`` or ``None``.
        :param rankby: see ``OpeniInterface().options('rankby')`` for valid values.
        :type rankby: ``list``, ``tuple`` or ``None``.
        :param article_type: see ``OpeniInterface().options('article_type')`` for valid values.
        :type article_type: ``list``, ``tuple`` or ``None``.
        :param subset: see ``OpeniInterface().options('subset')`` for valid values.
        :type subset: ``list``, ``tuple`` or ``None``.
        :param collection: see ``OpeniInterface().options('collection')`` for valid values.
        :type collection: ``list``, ``tuple`` or ``None``.
        :param fields: see ``OpeniInterface().options('fields')`` for valid values.
        :type fields: ``list``, ``tuple`` or ``None``.
        :param specialties: see ``OpeniInterface().options('specialties')`` for valid values.
        :type specialties: ``list``, ``tuple`` or ``None``.
        :param video: see ``OpeniInterface().options('video')`` for valid values. Defaults to ``None``.
        :type video: ``list``, ``tuple`` or ``None``.
        :param exclusions: one or both of: 'graphics', 'multipanel'.
                           Note: excluding 'multipanel' can result in images that ARE multipanel
                           being returned from Open-i API. For this reason, including 'multipanel'
                           is not currently recommended. Defaults to ['graphics'].
        :type exclusions: ``list``, ``tuple`` or ``None``
        :param print_results: if ``True``, print the number of search results.
        :type print_results: ``bool``
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

# ----------------------------------------------------------------------------------------------------------
# Pull Records from the NIH's Open-i API
# ----------------------------------------------------------------------------------------------------------

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

        self.results_df = None

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
            if isinstance(c, (list, tuple)) and c % self.sleep_main[0] == 0:
                if self.verbose:
                    print("\nSleeping for %s seconds..." % self.sleep_main[1])
                sleep(abs(self.sleep_main[1] + np.random.normal()))

            # Harvest
            harvested_data += self.openi_block_harvest(joined_url, bound, to_harvest)

            # Update counter
            c += 1

        # Return
        return harvested_data

    def _df_cleaning(self, data_frame):
        """

        :param data_frame:
        :return:
        """
        data_frame.columns = list(
            map(lambda x: camel_to_snake_case(x).replace("me_sh", "mesh"), data_frame.columns))

        # Make the type of Imaging technology type human-readable. ToDo: apply to the other image_modality.
        data_frame['image_modality_major'] = data_frame['image_modality_major'].map(
            lambda x: openi_image_type_params.get(cln(x).lower(), x), na_action='ignore'
        )

        # Look up the article type
        data_frame['article_type'] = data_frame['article_type'].map(
            lambda x: openi_article_type_params.get(cln(x).lower(), x), na_action='ignore'
        )

        # Replace 'Not Available' with NaN
        data_frame = data_frame.replace({'[nN]ot [aA]vailable.?': np.NaN}, regex=True)

        return data_frame

    def records_pull(self, search_query, to_harvest, total):
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

        # Convert to a DataFrame
        results_df = pd.DataFrame(harvest).fillna(np.NaN)

        # Clean and Add to attrs.
        self.results_df = self._df_cleaning(results_df)

        return self.results_df


        # ----------------------------------------------------------------------------------------------------------
        # Image Harvesting
        # ----------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------
# Construct Database
# ----------------------------------------------------------------------------------------------------------


class OpeniInterface(object):
    """

    Python Interface for the NIH's Open-i API.

    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    :param download_limit: max. number of results to download.
                           If ``None``, no limit will be imposed (not recommended). Defaults to 60.
    :type download_limit: ``int``
    :param img_sleep_time: time to sleep (in seconds) between requests for images. Noise is added on each call
                           by adding a value from a normal distrubition (with mean = 0, sd = 1). Defaults to 5 seconds.
    :type img_sleep_time: ``int`` or ``float``
    :param date_format: Defaults to ``'%d/%m/%Y'``.
    :type date_format: ``str``
    :param records_sleep_mini: Tuple of the form: (every x downloads, short peroid of time [seconds]). Defaults to (5, 1.5).
    :type records_sleep_mini: ``tuple``
    :param records_sleep_main: Tuple of the form: (every x downloads, long peroid of time [seconds]). Defaults to (50, 60).
    :type records_sleep_main: ``tuple``
    :param verbose: print additional details.
    :type verbose: ``bool``
    """

    def __init__(self
                 , cache_path=None
                 , download_limit=60
                 , img_sleep_time=1.5
                 , date_format='%d/%m/%Y'
                 , records_sleep_mini=(5, 1.5)
                 , records_sleep_main=(50, 60)  # ToDo: drop
                 , verbose=True):
        """

        Initialize the ``OpeniInterface()`` Class.

        """
        self._verbose = verbose
        self._root_url = 'https://openi.nlm.nih.gov'

        # Classes
        self._Search = _OpeniSearch()

        self.current_search_url = None
        self.current_search_total = None
        self._current_search_to_harvest = None

    def search(self
               , query
               , image_type=None
               , rankby=None
               , article_type=None
               , subset=None
               , collection=None
               , fields=None
               , specialties=None
               , video=None
               , exclusions=['graphics']
               , print_results=True):
        """

        Tool to generate a search term (URL) for the NIH's Open-i API.
        The computed term is stored as a class attribute (``INSTANCE.current_search_url``)

        :param query: a search term. ``None`` will be converted to an empty string.
        :type query: ``str`` or ``None``
        :param image_type: see ``OpeniInterface().options('image_type')`` for valid values.
        :type image_type: ``list``, ``tuple`` or ``None``.
        :param rankby: see ``OpeniInterface().options('rankby')`` for valid values.
        :type rankby: ``list``, ``tuple`` or ``None``.
        :param article_type: see ``OpeniInterface().options('article_type')`` for valid values.
        :type article_type: ``list``, ``tuple`` or ``None``.
        :param subset: see ``OpeniInterface().options('subset')`` for valid values.
        :type subset: ``list``, ``tuple`` or ``None``.
        :param collection: see ``OpeniInterface().options('collection')`` for valid values.
        :type collection: ``list``, ``tuple`` or ``None``.
        :param fields: see ``OpeniInterface().options('fields')`` for valid values.
        :type fields: ``list``, ``tuple`` or ``None``.
        :param specialties: see ``OpeniInterface().options('specialties')`` for valid values.
        :type specialties: ``list``, ``tuple`` or ``None``.
        :param video: see ``OpeniInterface().options('video')`` for valid values. Defaults to ``None``.
        :type video: ``list``, ``tuple`` or ``None``.
        :param exclusions: one or both of: 'graphics', 'multipanel'.
                           Note: excluding 'multipanel' can result in images that ARE multipanel
                           being returned from Open-i API. For this reason, including 'multipanel'
                           is not currently recommended. Defaults to ['graphics'].
        :type exclusions: ``list``, ``tuple`` or ``None``
        :param print_results: if ``True``, print the number of search results.
        :type print_results: ``bool``
        """
        # Note this simply wraps ``_OpeniSearch().search()``.
        self._Search.search(query=query,
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
        self.current_search_url = self._Search.current_search_url
        self.current_search_total = self._Search.current_search_total
        self._current_search_to_harvest = self._Search._current_search_to_harvest

    def _post_processing_text(self, data_frame):
        """

        :param data_frame:
        :return:
        """
        # snake_case from camelCase and lower.
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

        # Replace 'Not Available' with NaN
        data_frame = data_frame.replace({'[nN]ot [aA]vailable.?': np.NaN}, regex=True)

        return data_frame

    def pull(self):
        pass
























