# coding: utf-8

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
from warnings import warn
from itertools import chain
from datetime import datetime

# Biovida support tools
from biovida.support_tools.support_tools import cln, header, items_null
from biovida.support_tools._cache_management import package_cache_creator


class DiseaseOntInterface(object):
    """

    Python Interface for Harvesting the `Disease Ontology <http://disease-ontology.org/>`_ Database.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
                   Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    """

    def __init__(self, cache_path=None, verbose=True):
        self._verbose = verbose
        # Cache creation
        pcc = package_cache_creator(sub_dir='diagnostics',
                                    to_create=['disease_ontology'],
                                    cache_path=cache_path,
                                    verbose=verbose)
        self.root_path, self._created_disease_ont_dirs = pcc

        # The database itself
        self.disease_db = None
        self.db_date = None

    @staticmethod
    def _quote_value_parse(q):
        """

        Splits a string on the presence of quotes (") inside the string itself.
    
        :param q: a string containing quotes.
        :type q: ``str``
        :return: ``[text in quotes, text outside of quotes]``
        :rtype: ``list``
        """
        return list(map(cln, filter(None, q.split("\""))))
    
    def _def_url_parser(self, definition):
        """

        Separate the text of a disease definition from the list of reference URLs for that definition.
    
        :param definition: the raw text of the definition as represented in the Disease Ontology Database.
        :type definition: ``str``
        :return: a list of two ``tuple``s of the form:
                ``[('def', disease definition), ('def_urls', urls to def. sources)]``
        :rtype: ``list``
        """
        # Note: this method automatically replaces the underscores in the definition quotes.
        # This seems to be valid currently as this shortcut seems to invariably produce the same
        # result as would be obtained by following the list of mappings at the top of the .obo
        # database file. *This could change*.
        if definition.count("\"") != 2 or definition.count("[") != 1 or definition.count("]") != 1:
            return [("def", definition), ("def_urls", np.NaN)]

        # Separate the quote from the list of URLS
        parsed_definition = self._quote_value_parse(definition)
    
        # Extract the list of urls
        urls = parsed_definition[1].lower().replace("url:", "").replace("[", "").replace("]", "").split(", ")
    
        # Remove escape for the colon in the urls
        cleaned_urls = [u.replace("\:/", ":/") for u in urls]
    
        # Return the quote and the urls as separate entities
        return [("def", parsed_definition[0].replace("_", " ")), ("def_urls", cleaned_urls)]

    @staticmethod
    def _is_a_parser(is_a):
        """

        Separates the DOID from the text of the entry in an 'is_a' entry.

        :param is_a: a value corresponding to a term entry 'is_a' in the Disease Ontology database
        :type is_a: ``str``
        :return: list of tuples of the form:
                `` [("is_a", text of entry), ("is_a_doid", DOID number of entry)]``
        :rtype: ``list``
        """
        if " ! " not in is_a:
            return is_a
        parse_input = cln(is_a).split(" ! ")
        return [("is_a", parse_input[1]), ("is_a_doid", parse_input[0].upper().replace("DOID:", ""))]
    
    def _value_parser(self, k, v):
        """

        Parses the ``v`` evolved inside ``DiseaseOntInterface()._parsed_term_to_dict()`` based on ``k``.
    
        :param k: the k (key) passed inside of ``DiseaseOntInterface()._parsed_term_to_dict`` .
        :type k: ``str``
        :param v: the v (value) passed inside of ``DiseaseOntInterface()._parsed_term_to_dict``.
        :type v: ``str``
        :return: one of:

        - ``[('def', disease definition), ('def_urls', urls to def. sources)]``

        - ``[(KEY, value.replace("DOID:", ""))]``

        - ``[(KEY, information within quotes in v), (KEY_FLAG, list information in v)]``*

        - ``[(KEY, value)]``

        *Example: "'the quick brown fox' EXACT [info1, info2, info3]", where 'EXACT' is the FLAG.
        Here:

            - "information within quotes in v" = 'the quick brown fox'.

            - "list information in v": "[info1, info2, info3]" (string) --> ['info1', 'info2', 'info3']  (python list).

        :rtype: ``list``
        """
        if k == 'def':
            return self._def_url_parser(v)
        elif k == 'is_a':
            return self._is_a_parser(v)
        elif k in ('id', 'alt_id'):
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

        Parses an individual "[Term]" in the database and recasts it as a dictionary.
    
        :param parsed_term: a list of the form [[KEY, VALUE]...] as described
        :type parsed_term: ``list``
        :return: ``parsed_term`` converted into a dictionary where the information as well as a list of the keys 
                (future column names) which contain lists.
        :rtype: ``tuple``
        """
        d = dict()
        keys_with_lists = set()
        for (k, v) in parsed_term:
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
        
        This method splits a "[Term]" on line breaks ("\n"), the resultant list
        is then converted to a list of lists (where the sublists are invariably of length `2`),
        by splitting the elements into keys and values, e.g., ``['id', '000000']``.
        This list of lists is passed to ``DiseaseOntInterface()._parsed_term_to_dict()`` where it
        is converted into a dictionary.
        
        :param term: a single "[Term]" from the Disease Ontology Database.
        :type term: ``str``
        :return: a term which as been parsed by ``DiseaseOntInterface()._parsed_term_to_dict()``.
        :rtype: ``dict``
        """
        # Split the term on line breaks
        split_term = filter(None, cln(term).split("\n"))
    
        # Split each element in `term` on the ": " pattern.
        parsed_term = [i.split(": ", 1) for i in split_term]
    
        # Convert to a dict and return
        return self._parsed_term_to_dict(parsed_term)

    @staticmethod
    def _do_df_cleaner(data_frame, columns_with_lists):
        """

        This method cleans the final Disease Ontology database in the following ways:

        - converts columns which contains lists to strings.

        - lowers all strings in the following columns: 'name', 'synonym', 'subset' and 'is_a'.

        - converts the 'true' string in the 'is_obsolete' column to an actual python boolean ``True``.
    
        :param data_frame: the dataframe evolved in the ``DiseaseOntInterface()._harvest_engine()`` method.
        :type data_frame: ``Pandas DataFrame``
        :param columns_with_lists: a 'list' of columns in the dataframe which contain lists.
        :type columns_with_lists: ``iterable``
        :return: a dataframe with the above mentioned cleaning steps taken.
        :rtype: ``Pandas DataFrame``
        """
        # Homogenize columns with lists
        for c in columns_with_lists:
            data_frame[c] = data_frame[c].map(lambda x: "; ".join(x) if isinstance(x, list) else x,
                                              na_action='ignore')
    
        # Lower columns to make it easier to match in the future
        for c in ('name', 'synonym', 'subset', 'is_a'):
            data_frame[c] = data_frame[c].map(lambda x: str(x).lower(), na_action='ignore')
    
        # Convert 'true' in the 'is_obsolete' column to an actual python boolean ``True``.
        data_frame['is_obsolete'] = data_frame['is_obsolete'].map(
            lambda x: True if not items_null(x) and str(x).lower().strip() == 'true' else x,
            na_action='ignore')
    
        return data_frame

    def _extract_date_version(self, first_parsed_by_term):
        """

        Extracts the date the Disease Ontology data was created.

        :param first_parsed_by_term: the first element in the list obtained by splitting the database on '[Term]'.
        :type first_parsed_by_term: ``str``
        """
        try:
            extracted_date = re.search('data-version: (.*)\n', first_parsed_by_term).group(1)
            extracted_date_cleaned = "".join((i for i in extracted_date if i.isdigit() or i == "-")).strip()
            self.db_date = datetime.strptime(extracted_date_cleaned, "%Y-%m-%d")
        except:
            warn("\nCould not extract the date on which the Disease Ontology database was generated.")

    def _harvest_engine(self, disease_ontology_db_url, **kwargs):
        """

        This method Harvests and orchestrates and conversion of the Disease Ontology Database
        from '.obo' format to a Pandas DataFrame.

        :param disease_ontology_db_url: see: ``DiseaseOntInterface().pull()``.
        :type disease_ontology_db_url: ``str``
        """
        if 'obo_file' in kwargs:
            obo_file = kwargs.get('obo_file')
        else:
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
    
        # Extract keys (future column names) which contain lists
        keys_with_lists = filter(None, (i[1] for i in fully_parsed_terms))
    
        # Compress `keys_with_lists` to uniques.
        columns_with_lists = set(chain(*keys_with_lists))
    
        # Convert to a DataFrame, Clean and Return
        self.disease_db = self._do_df_cleaner(pd.DataFrame(list_of_dicts), columns_with_lists)

    def pull(self, download_override=False, disease_ontology_db_url='http://purl.obolibrary.org/obo/doid.obo'):
        """

        Pull (i.e., download) the Disease Ontology Database.

        Notes:

        - if a database is already cached, it will be used instead of downloading (use ``download_override`` to override).

        - multiple values are separated by semicolons followed by a space, i.e., "; ".

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
                                  Defaults to ``False``.
        :type download_override: ``bool``
        :param disease_ontology_db_url: URL to the disease ontology database in '.obo' format.
                                        Defaults to 'http://purl.obolibrary.org/obo/doid.obo'.
        :type disease_ontology_db_url: ``str``
        :return: the Disease Ontology database as a DataFrame.
        :rtype: ``Pandas DataFrame``
        """
        save_path = os.path.join(self._created_disease_ont_dirs['disease_ontology'], "disease_ontology_db")
        db_path = "{0}.p".format(save_path)
        support_path = "{0}_support.p".format(save_path)

        if not os.path.isfile(db_path) or download_override:
            if self._verbose:
                header("Downloading Disease Ontology Database... ")
            self._harvest_engine(disease_ontology_db_url)
            self.disease_db.to_pickle(db_path)
            pickle.dump(self.db_date, open(support_path, "wb"))
        elif 'dataframe' not in str(type(self.disease_db)).lower():
            with open(support_path, "rb") as f:
                self.db_date = pickle.load(f)
            self.disease_db = pd.read_pickle(db_path)

        return self.disease_db
