"""

    Disease Ontology
    ~~~~~~~~~~~~~~~~

"""
import re
import requests
import numpy as np
import pandas as pd
from pprint import pprint
from itertools import chain
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import items_null


def quote_value_parse(q):
    """

    :param q:
    :return:
    """
    return list(map(cln, filter(None, q.split("\""))))


def def_url_parser(definition):
    """

    :param definition:
    :return:
    """
    if definition.count("\"") != 2 or definition.count("[") != 1 or definition.count("]") != 1:
        return (("def", definition), ("def_urls", np.NaN))

    # Seperate the quote from the list of URLS
    parsed_definition = quote_value_parse(definition)

    # Extract the list of urls
    urls = parsed_definition[1].lower().replace("url:", "").replace("[", "").replace("]", "").split(", ")

    # Remove escape for the colon in the urls
    cleaned_urls = [u.replace("\:/", ":/") for u in urls]

    # Return the quote and the urls as seperate entities
    return [("def", parsed_definition[0]), ("def_urls", cleaned_urls)]


def is_a_parser(is_a):
    """

    :param is_a:
    :return:
    """
    if " ! " not in is_a:
        return is_a
    parse_input = cln(is_a).split(" ! ")
    return [("is_a", parse_input[1]), ("is_a_doid", parse_input[0].upper().replace("DOID:", ""))]


def value_parser(k, v):
    """

    :param k:
    :param v:
    :return:
    """
    if k == 'def':
        return def_url_parser(v)
    elif k == 'is_a':
        return is_a_parser(v)
    elif k in ['id', 'alt_id']:
        return [(k, v.upper().replace("DOID:", ""))]
    elif v.count("\"") == 2 and v.count("[") == 1 and v.count("]") == 1:
        # Split the true quote and the 'flags' (e.g., 'EXACT').
        parsed_v = quote_value_parse(v)
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


def parsed_term_to_dict(parsed_term):
    """

    :param parsed_term:
    :return:
    """
    d = dict()
    keys_with_lists = set()
    for (k, v) in parsed_term:
        # Split values by the presence of quotes.
        parsed = value_parser(k, v=cln(v))
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
                        d[kp] = vp
                else:
                    keys_with_lists.add(kp)
                    d[kp] = [d[kp], vp]

    return d, keys_with_lists


def do_term_parser(term):
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
    return parsed_term_to_dict(parsed_term)


def do_df_cleaner(data_frame, columns_with_lists):
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


def pull(disease_ontology_db_url='http://purl.obolibrary.org/obo/doid.obo'):
    """

    :param disease_ontology_db_url: URL to the
    :type disease_ontology_db_url: ``str``
    :return:
    """
    # Open the file and discard [Typedef] information at the end of the file.
    obo_file = requests.get(disease_ontology_db_url, stream=True).text.split("[Typedef]")[0]

    # Parse the file by spliting on [Term].
    parsed_by_term = obo_file.split("[Term]\n")

    # pprint(parsed_by_term[0]) -- Use.

    # Convert to a list of dicts
    fully_parsed_terms = [do_term_parser(term) for term in parsed_by_term[1:]]

    # Extract the dicts
    list_of_dicts = [i[0] for i in fully_parsed_terms]

    # Extract keys (future columns) which contain lists
    keys_with_lists = filter(None, (i[1] for i in fully_parsed_terms))

    # Compress `keys_with_lists` to uniques.
    columns_with_lists = set(chain(*keys_with_lists))

    # Convert to a DataFrame; Clean and Return
    return do_df_cleaner(pd.DataFrame(list_of_dicts), columns_with_lists)












































































