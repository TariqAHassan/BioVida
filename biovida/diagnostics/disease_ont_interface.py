"""

    Disease Ontology Interface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import re
import pickle
import requests
import numpy as np
import pandas as pd
from itertools import chain
from datetime import datetime
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import items_null
from biovida.support_tools._cache_management import _package_cache_creator


class DiseaseOntInterface(object):
    """

    Python Interface for Harvesting the Complete Disease Ontology Database.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
                   Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    """

    def __init__(self, cache_path=None, verbose=True):
        self._verbose = verbose
        # Cache creation
        pcc = _package_cache_creator(sub_dir='diagnostics', to_create=['disease_ontology'], cache_path=cache_path)
        self.root_path, self._created_disease_ont_dirs = pcc

        # The database itself
        self.disease_db = None
        self.db_date = None

    def _quote_value_parse(self, q):
        """
    
        :param q:
        :return:
        """
        return list(map(cln, filter(None, q.split("\""))))
    
    def _def_url_parser(self, definition):
        """
    
        :param definition:
        :return:
        """
        if definition.count("\"") != 2 or definition.count("[") != 1 or definition.count("]") != 1:
            return [("def", definition), ("def_urls", np.NaN)]
    
        # Separate the quote from the list of URLS
        parsed_definition = self._quote_value_parse(definition)
    
        # Extract the list of urls
        urls = parsed_definition[1].lower().replace("url:", "").replace("[", "").replace("]", "").split(", ")
    
        # Remove escape for the colon in the urls
        cleaned_urls = [u.replace("\:/", ":/") for u in urls]
    
        # Return the quote and the urls as seperate entities
        return [("def", parsed_definition[0]), ("def_urls", cleaned_urls)]
    
    def _is_a_parser(self, is_a):
        """
    
        :param is_a:
        :return:
        """
        if " ! " not in is_a:
            return is_a
        parse_input = cln(is_a).split(" ! ")
        return [("is_a", parse_input[1]), ("is_a_doid", parse_input[0].upper().replace("DOID:", ""))]
    
    def _value_parser(self, k, v):
        """
    
        :param k:
        :param v:
        :return:
        """
        if k == 'def':
            return self._def_url_parser(v)
        elif k == 'is_a':
            return self._is_a_parser(v)
        elif k in ['id', 'alt_id']:
            return [(k, v.upper().replace("DOID:", ""))]
        elif v.count("\"") == 2 and v.count("[") == 1 and v.count("]") == 1:
            # Split the true quote and the 'flags' (e.g., 'EXACT').
            parsed_v = self._quote_value_parse(v)
            # Split the flag and its corresponding list
            additional_v = re.split(r'\s(?=\[)', parsed_v[1])
            # Clean the flag list
            cleaned_flag = map(cln, additional_v[1].replace("[", "").replace("]", "").split(", "))
            # Filter the flag list
            related_info = list(filter(None, cleaned_flag))
            # Return the (key, quote) and the (key_flag, info).
            return [(k, parsed_v[0]), ("{0}_{1}".format(k, additional_v[0].lower()), related_info)]
        else:
            return [(k, v)]
    
    def _parsed_term_to_dict(self, parsed_term):
        """
    
        :param parsed_term:
        :return:
        """
        d = dict()
        keys_with_lists = set()
        for (k, v) in parsed_term:
            # Split values by the presence of quotes.
            parsed = self._value_parser(k, v=cln(v))
            for (kp, vp) in parsed:
                if kp not in d:
                    if isinstance(vp, list):
                        if len(vp):
                            keys_with_lists.add(kp)
                            d[kp] = vp
                        else:
                            d[kp] = np.NaN
                    else:
                        d[kp] = vp
                elif kp in d:
                    if isinstance(d[kp], list):
                        d[kp] += vp if isinstance(vp, list) else [vp]
                        keys_with_lists.add(kp)
                    elif items_null(d[kp]):
                        # In short, if the current value is NaN, replace with it with `vp` if
                        # and only if it is a list with nonzero length, otherwise leave as a NaN.
                        if isinstance(vp, list) and len(vp):
                            keys_with_lists.add(kp)
                            d[kp] = vp
                    else:
                        keys_with_lists.add(kp)
                        d[kp] = [d[kp], vp]
    
        return d, keys_with_lists
    
    def _do_term_parser(self, term):
        """
    
        :param term:
        :type term:
        :return:
        """
        # Split the term on line breaks
        split_term = list(filter(None, cln(term).split("\n")))
    
        # Split each element in `term` on the ": " pattern.
        parsed_term = [i.split(": ", 1) for i in split_term]
    
        # Convert to a dict and return
        return self._parsed_term_to_dict(parsed_term)
    
    def _do_df_cleaner(self, data_frame, columns_with_lists):
        """
    
        :param data_frame:
        :param columns_with_lists:
        :return:
        """
        # Homogenize columns with lists
        for c in columns_with_lists:
            data_frame[c] = data_frame[c].map(lambda x: "; ".join(x) if isinstance(x, list) else x, na_action='ignore')
    
        # Lower columns to make it easier to match in the future
        for c in ['name', 'synonym', 'subset', 'is_a']:
            data_frame[c] = data_frame[c].map(lambda x: x.lower(), na_action='ignore')
    
        # Fix 'is_obsolete'
        data_frame['is_obsolete'] = data_frame['is_obsolete'].map(
            lambda x: True if not items_null(x) and x.lower().strip() == 'true' else x, na_action='ignore'
        )
    
        return data_frame

    def _extract_date_version(self, first_parsed_by_term):
        """

        :param first_parsed_by_term:
        :return:
        """
        extracted_date = re.search('data-version: (.*)\n', first_parsed_by_term).group(1)
        extracted_date_cleaned = "".join((i for i in extracted_date if i.isdigit() or i == "-")).strip()
        self.db_date = datetime.strptime(extracted_date_cleaned, "%Y-%m-%d")

    def _harvest(self, disease_ontology_db_url):
        """
    
        :param disease_ontology_db_url: see: ``DiseaseOntInterface().pull()``.
        :type disease_ontology_db_url: ``str``
        """
        # Open the file and discard [Typedef] information at the end of the file.
        obo_file = requests.get(disease_ontology_db_url, stream=True).text.split("[Typedef]")[0]
    
        # Parse the file by splitting on [Term].
        parsed_by_term = obo_file.split("[Term]\n")
    
        # Extract the date
        self._extract_date_version(parsed_by_term[0])

        # Convert to a list of dicts
        fully_parsed_terms = [self._do_term_parser(term) for term in parsed_by_term[1:]]
    
        # Extract the dicts
        list_of_dicts = [i[0] for i in fully_parsed_terms]
    
        # Extract keys (future columns) which contain lists
        keys_with_lists = filter(None, (i[1] for i in fully_parsed_terms))
    
        # Compress `keys_with_lists` to uniques.
        columns_with_lists = set(chain(*keys_with_lists))
    
        # Convert to a DataFrame, Clean and Return
        self.disease_db = self._do_df_cleaner(pd.DataFrame(list_of_dicts), columns_with_lists)

    def pull(self, download_override=False, disease_ontology_db_url='http://purl.obolibrary.org/obo/doid.obo'):
        """

        Pull (i.e., download) the ODisease Ontology Database.

        Note: if a database is already cached, it will be used instead of downloading
        (the `download_override` argument can be used override this behaviour).

        :param download_override: If True, override any existing database currently cached and download a new one.
                                  Defaults to False.
        :type download_override: ``bool``
        :param disease_ontology_db_url: URL to the disease ontology .obo database.
                                        Defaults to 'http://purl.obolibrary.org/obo/doid.obo'.
        :type disease_ontology_db_url: ``str``
        :return:
        """
        save_path = os.path.join(self._created_disease_ont_dirs['disease_ontology'], "disease_ontology_db")
        db_path = "{0}.csv".format(save_path)
        support_path = "{0}_support.p".format(save_path)

        if not os.path.isfile(db_path) or download_override:
            if self._verbose:
                print("Downloading Database...")
            self._harvest(disease_ontology_db_url)
            self.disease_db.to_csv(db_path, index=False)
            pickle.dump(self.db_date, open(support_path, "wb"))
        elif 'dataframe' not in str(type(self.disease_db)).lower() or 'datetime' not in str(type(self.db_date)).lower():
            self.db_date = pickle.load(open(support_path, "rb"))
            self.disease_db = pd.read_csv(db_path)

        return self.disease_db













































