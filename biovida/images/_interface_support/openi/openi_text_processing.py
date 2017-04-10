# coding: utf-8

"""

    Tools to Clean Raw Open-i Text
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter

# General Support tools
from biovida.support_tools.support_tools import cln, unescape, remove_html_bullet_points

# Image Support Tools
from biovida.images._image_tools import NoResultsFound

# General Support Tools
from biovida.support_tools.support_tools import tqdm, camel_to_snake_case, reverse_comma, order_set

# Open-i API Parameters Information
from biovida.images._interface_support.openi.openi_parameters import (openi_image_type_params,
                                                                      openi_article_type_params,
                                                                      openi_image_type_modality_full)

from biovida.images._interface_support.openi._openi_image_id_processing import image_id_short_gen
from biovida.images._interface_support.openi._openi_text_feature_extraction import (CLINICAL_ARTICLE_TYPES,
                                                                                    feature_extract)

# Other BioVida APIs
from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface


# ----------------------------------------------------------------------------------------------------------
# Abstract Cleaning
# ----------------------------------------------------------------------------------------------------------


def abstract_cleaner(abstract):
    """

    Clean ``abstract`` by converting the HTML into standard text.

    :param abstract: an abstract obtained via. the Open-i API.
    :type abstract: ``str``
    :return: cleaned ``abstract``
    :rtype: ``str``
    """
    if not isinstance(abstract, str):
        return np.NaN

    soup = BeautifulSoup(remove_html_bullet_points(abstract).replace("<b>", ". "), 'lxml')
    cleaned = soup.text.replace(" ; ", " ").replace("..", ".").replace(".;", ";").strip()
    return cleaned.strip(".") + "."


# ----------------------------------------------------------------------------------------------------------
# Problems Cleaning
# ----------------------------------------------------------------------------------------------------------


def problems_cleaner(problems):
    """
    
    Clean strings in the 'problems'
    column by:
    
    - removing duplicates
    - reversing items with commas
    
    :param problems: an item from the 'problems' column
    :return: cleaned ``problems``
    :rtype: ``str``
    
    :Example:
    
    >>> problems_cleaner("Lung; Fractures, Bone")
    ...
    'lung; bone fractures'
    
    """
    if not isinstance(problems, str):
        return problems

    cleaned = order_set(problems.lower().split(";"))
    return "; ".join([reverse_comma(s=cln(i)) for i in cleaned])


# ----------------------------------------------------------------------------------------------------------
# Article Type Lookup
# ----------------------------------------------------------------------------------------------------------


def article_type_lookup(article_type_abbrev):
    """

    Lookup the full article type from the shorthand used by the Open-i API.

    :param article_type_abbrev: values as passed via. ``data_frame['article_type'].map(...)``.
    :return: the full article type name.
    :rtype: ``str``
    """
    # Note: It appears that, at the time of writing (March 16, 2017), Open-i has switched
    # to transforming this data on their end. However, this function remains as a safeguard.
    if not isinstance(article_type_abbrev, str):
        return article_type_abbrev

    return openi_article_type_params.get(cln(article_type_abbrev).lower(), article_type_abbrev)


# ----------------------------------------------------------------------------------------------------------
# Handle Missing Columns
# ----------------------------------------------------------------------------------------------------------


def _df_add_missing_columns(data_frame):
    """

    Add 'license_type', 'license_url' and 'image_caption_concepts' columns
    if they do not exist in ``data_frame``

    :param data_frame: the dataframe evolved inside ``openi_raw_extract_and_clean``.
    :type data_frame: ``Pandas DataFrame``
    :return: see description.
    :rtype: ``Pandas DataFrame``
    """
    # Handle cases where some searches (e.g., collection='pubmed')
    # Open-i does not return these columns (not clear why...).
    to_check = ['license_type', 'image_caption_concepts', 'pub_med_url', 'license_url',
                'impression', 'note', 'medpix_article_id', 'medpix_figure_id', 'medpix_image_url']

    for c in to_check:
        if c not in data_frame.columns:
            data_frame[c] = [np.NaN] * data_frame.shape[0]
    return data_frame


# ----------------------------------------------------------------------------------------------------------
# Tool to Limit to Clinical Cases
# ----------------------------------------------------------------------------------------------------------


def _apply_clinical_case_only(data_frame):
    """

    Remove records (dataframe rows) which are not of clinical encounters.

    Note: this is here, and not simply part of defining a search with ``openi_interface()._OpeniImages().search()``
    because Open-i API's 'article_type' (&at) parameter does not have an 'encounter' option.

    :param data_frame: the ``data_frame`` as evolved in ``openi_raw_extract_and_clean()``.
    :type data_frame: ``Pandas DataFrame``
    :return: see description.
    :rtype: ``Pandas DataFrame``
    """
    def test(article_type):
        if isinstance(article_type, str) and article_type in CLINICAL_ARTICLE_TYPES:
            return True
        else:
            return False

    data_frame = data_frame[data_frame['article_type'].map(test)].reset_index(drop=True)

    if data_frame.shape[0] == 0:
        raise NoResultsFound("\nNo results remained after the `clinical_cases_only=True` restriction was applied.\n"
                             "Consider setting `pull()`'s `clinical_cases_only` parameter to `False`.")
    return data_frame


# ----------------------------------------------------------------------------------------------------------
# Make Hashable
# ----------------------------------------------------------------------------------------------------------


def _iterable_cleaner(iterable):
    """

    Clean terms in an iterable and return as a tuple.

    :param iterable: a list of terms.
    :type iterable: ``tuple`` or ``list``
    :return: a cleaned tuple of strings.
    :rtype: ``tuple`` or ``type(iterable)``
    """
    if isinstance(iterable, (list, tuple)):
        return tuple([cln(unescape(i)) if isinstance(i, str) else i for i in iterable])
    else:
        return iterable


def _df_make_hashable(data_frame):
    """

    Ensure the records dataframe can be hashed (i.e., ensure ``pandas.DataFrame.drop_duplicates`` does not fail).

    :param data_frame: the dataframe evolved inside ``openi_raw_extract_and_clean``.
    :type data_frame: ``Pandas DataFrame``
    :return: ``data_frame`` corrected such that all columns considered can be hashed.
    :rtype: ``Pandas DataFrame``
    """
    # Escape HTML elements in the 'image_caption' and 'image_mention' columns.
    for c in ('image_caption', 'image_mention'):
        data_frame[c] = data_frame[c].map(lambda x: cln(unescape(x)) if isinstance(x, str) else x, na_action='ignore')

    # Clean mesh terms
    for c in ('mesh_major', 'mesh_minor', 'image_caption_concepts'):
        data_frame[c] = data_frame[c].map(_iterable_cleaner, na_action='ignore')

    # Convert all elements in the 'license_type' and 'license_url' columns to string.
    for c in ('license_type', 'license_url'):
        data_frame[c] = data_frame[c].map(lambda x: "; ".join(map(str, x)) if isinstance(x, (list, tuple)) else x,
                                          na_action='ignore')

    return data_frame


# ----------------------------------------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------------------------------------


def _data_frame_fill_nan(data_frame):
    """

    Replace terms that are synonymous with NA with NaNs.

    :param data_frame: the dataframe evolved inside ``openi_raw_extract_and_clean``.
    :type data_frame: ``Pandas DataFrame``
    :return: ``data_frame`` with terms that are synonymous with NA replaced with NaNs.
    :rtype:  ``Pandas DataFrame``
    """
    to_nan = ('[nN]ot [aA]vailable.?', '[Nn]one.?', '[Nn]/[Aa]', '[Nn][Aa]',
              '[Ii][Nn] [Pp][Rr][Oo][Gg][Rr][Ee][Ss][Ss].?')
    # Anchor (i.e., exact matches are required, e.g., "NA" --> NaN, but "here we see NA" will not be converted).
    anchored_to_nan = map(lambda x: "^{0}$".format(x), to_nan)
    data_frame = data_frame.replace(dict.fromkeys(anchored_to_nan, np.NaN), regex=True)

    # Replace the 'replace this - ' placeholder with NaN
    data_frame['image_caption'] = data_frame['image_caption'].map(
        lambda x: np.NaN if isinstance(x, str) and cln(x).lower().startswith('replace this - ') else x,
        na_action='ignore')

    return data_frame.fillna(np.NaN)


def _data_frame_clean(data_frame, verbose):
    """

    Clean the text information.

    :param data_frame: the dataframe evolved inside ``openi_raw_extract_and_clean``.
    :type data_frame: ``Pandas DataFrame``
    :param verbose: if ``True`` print additional details.
    :type verbose: ``bool``
    :return: cleaned ``data_frame``.
    :rtype:  ``Pandas DataFrame``
    """
    # Clean the abstract
    data_frame['abstract'] = data_frame['abstract'].map(abstract_cleaner)

    # Clean the 'image_caption'
    data_frame['image_caption'] = data_frame['image_caption'].map(
        lambda x: cln(BeautifulSoup(x, 'lxml').text) if isinstance(x, str) else x, na_action='ignore')

    # Add the full name for modalities (before the 'image_modality_major' values are altered below).
    data_frame['modality_full'] = data_frame['image_modality_major'].map(
        lambda x: openi_image_type_modality_full.get(cln(x).lower(), x), na_action='ignore')

    # Make the type of Imaging technology type human-readable.
    data_frame['image_modality_major'] = data_frame['image_modality_major'].map(
        lambda x: openi_image_type_params.get(cln(x).lower(), x), na_action='ignore')

    # Label the number of instance of repeating 'uid's.
    data_frame = image_id_short_gen(data_frame)

    # Replace missing Values with with NaN and Return
    return _data_frame_fill_nan(data_frame)


# ----------------------------------------------------------------------------------------------------------
# Image Caption Count Analysis
# ----------------------------------------------------------------------------------------------------------


def _unique_image_caption_dict_gen(data_frame, verbose):
    """

    Generate a dictionary of the form:
    ``{'uid': {'image caption goes here': unique (bool), ...}, ...}``

    :param data_frame: as evolved in ``openi_raw_extract_and_clean()``
    :type data_frame: ``Pandas DataFrame``
    :param verbose: if ``True`` print additional details.
    :type verbose: ``bool``
    :return: see description
    :rtype: ``dict``
    """
    large_data_frame = 25000  # after this number of rows, processing time is notable.

    def counter_wrapper(l):
        l_cleaned = filter(lambda x: isinstance(x, str) and len(cln(x)), l)
        count_dict = dict(Counter(l_cleaned))
        return {k: v == 1 for k, v in count_dict.items()} if count_dict else None

    def counter_wrapper_apply(row):
        return counter_wrapper(row['image_caption'].tolist())

    if len(data_frame) >= large_data_frame and verbose:
        print("\nAnalyzing Image Caption Frequency...")

    d = data_frame.groupby('uid').apply(counter_wrapper_apply).to_dict()

    return {k: v for k, v in d.items() if v is not None}


# ----------------------------------------------------------------------------------------------------------
# Outward Facing Tool
# ----------------------------------------------------------------------------------------------------------


def openi_raw_extract_and_clean(data_frame, clinical_cases_only, verbose, cache_path):
    """

    Extract features from, and clean text of, ``data_frame``.

    :param data_frame: the dataframe evolved inside ``biovida.images.openi_interface._OpeniRecords().records_pull()``.
    :rtype data_frame: ``Pandas DataFrame``
    :param clinical_cases_only: if ``True`` require that the data harvested is of a clinical case. Specifically,
                                this parameter requires that 'article_type' is one of: 'encounter', 'case_report'.
                                Defaults to ``True``.
    :type clinical_cases_only: ``bool``
    :param verbose: if ``True`` print additional details.
    :type verbose: ``bool``
    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str``
    :return: see description.
    :rtype:  ``Pandas DataFrame``
    """
    data_frame.columns = list(map(lambda x: camel_to_snake_case(x).replace("me_sh", "mesh"), data_frame.columns))
    data_frame = _df_add_missing_columns(data_frame)

    data_frame['article_type'] = data_frame['article_type'].map(article_type_lookup, na_action='ignore')
    data_frame['problems'] = data_frame['problems'].map(problems_cleaner, na_action='ignore')

    if clinical_cases_only:
        data_frame = _apply_clinical_case_only(data_frame)

    # Ensure the dataframe can be hashed (i.e., ensure pandas.DataFrame.drop_duplicates does not fail).
    data_frame = _df_make_hashable(data_frame)

    list_of_diseases = DiseaseOntInterface(cache_path=cache_path, verbose=verbose).pull()['name'].tolist()
    unique_image_caption_dict = _unique_image_caption_dict_gen(data_frame=data_frame, verbose=verbose)

    def feature_extract_wrapper(row):
        caption_unique_bool = unique_image_caption_dict.get(row['uid'], {}).get(row['image_caption'], False)
        return feature_extract(row, list_of_diseases=list_of_diseases, image_caption_unique=caption_unique_bool)

    # Run Feature Extracting Tool and Join with `data_frame`.
    extract = [feature_extract_wrapper(row) for _, row in tqdm(data_frame.iterrows(), desc='Processing Records',
                                                               total=len(data_frame), disable=not verbose)]
    extract_df = pd.DataFrame(extract)

    if not any(c in data_frame.columns for c in extract_df.columns):
        data_frame = data_frame.join(extract_df, how='left')
    else:
        for c in extract_df.columns:
            data_frame[c] = extract_df[c]

    return _data_frame_clean(data_frame, verbose=verbose)
