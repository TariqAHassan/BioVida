"""

    Tools to Clean Raw Open-i Text
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# General Support tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import unescape
from biovida.support_tools.support_tools import remove_from_head_tail
from biovida.support_tools.support_tools import remove_html_bullet_points

# Image Support Tools
from biovida.images._image_tools import NoResultsFound
from biovida.images._image_tools import resetting_label

# General Support Tools
from biovida.support_tools.support_tools import camel_to_snake_case

# Open-i API Parameters Information
from biovida.images._interface_support.openi.openi_parameters import openi_image_type_params
from biovida.images._interface_support.openi.openi_parameters import openi_article_type_params
from biovida.images._interface_support.openi.openi_parameters import openi_image_type_modality_full

# Tools for Text Feature Extraction
from biovida.images._interface_support.openi.openi_text_feature_extraction import feature_extract

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
    cleaned = soup.text.replace(" ; ", " ").replace("..", ".").replace(".;", ";")
    return remove_from_head_tail(cleaned, char=".") + "."


# ----------------------------------------------------------------------------------------------------------
# Article Type Lookup
# ----------------------------------------------------------------------------------------------------------


def article_type_lookup(article_type_abbrev):
    """

    Lookup the full article type from the shorthand used by the Open-i API.

    :param article_type_abbrev: values as passed via. ``data_frame['article_type'].map(...)`` in
    :return: the full article type name.
    :rtype: ``str``
    """
    if not isinstance(article_type_abbrev, str):
        return article_type_abbrev

    rslt = openi_article_type_params.get(cln(article_type_abbrev).lower(), None)
    return cln(rslt.replace("_", " ")) if isinstance(rslt, str) else article_type_abbrev


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
    for c in ('license_type', 'license_url', 'image_caption_concepts'):
        if c not in data_frame.columns:
            data_frame[c] = [np.NaN] * data_frame.shape[0]
    return data_frame


# ----------------------------------------------------------------------------------------------------------
# Tool to Limit to Clinical Cases
# ----------------------------------------------------------------------------------------------------------


def _apply_clinical_case_only(data_frame):
    """

    Remove records (dataframe rows) which are not of clinical encounters.

    Note: this is here, and not in ``openi_interface()._OpeniImages().pull_images()`` because
    Open-i API's 'article_type (&at) parameter does not have an 'encounter' option
    (which it probably should...).

    :param data_frame: the ``data_frame`` as evolved in ``openi_raw_extract_and_clean()``.
    :type data_frame: ``Pandas DataFrame``
    :return: see description.
    :rtype: ``Pandas DataFrame``
    """
    clinical_article_types = ('encounter', 'case report')

    def test(article_type):
        if isinstance(article_type, str) and article_type in clinical_article_types:
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

    Ensure the records dataframe can be hashed (i.e., ensure pandas.DataFrame.drop_duplicates does not fail).

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


def _df_fill_nan(data_frame):
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


def _df_clean(data_frame):
    """

    Clean the text information.

    :param data_frame: the dataframe evolved inside ``openi_raw_extract_and_clean``.
    :type data_frame: ``Pandas DataFrame``
    :return: cleaned ``data_frame``.
    :rtype:  ``Pandas DataFrame``
    """
    # Clean the abstract
    data_frame['abstract'] = data_frame['abstract'].map(abstract_cleaner)

    # Add the full name for modalities (before the 'image_modality_major' values are altered below).
    data_frame['modality_full'] = data_frame['image_modality_major'].map(
        lambda x: openi_image_type_modality_full.get(cln(x).lower(), x), na_action='ignore')

    # Make the type of Imaging technology type human-readable. ToDo: apply to the other image_modality.
    data_frame['image_modality_major'] = data_frame['image_modality_major'].map(
        lambda x: openi_image_type_params.get(cln(x).lower(), x), na_action='ignore')

    # Label the number of instance of repeating 'uid's.
    data_frame['uid_instance'] = resetting_label(data_frame['uid'].tolist())

    # Replace missing Values with with NaN and Return
    return _df_fill_nan(data_frame)


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
    :param verbose: print additional details.
    :type verbose: ``bool``
    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str``
    :return: see description.
    :rtype:  ``Pandas DataFrame``
    """
    # Convert column names to snake_case
    data_frame.columns = list(map(lambda x: camel_to_snake_case(x).replace("me_sh", "mesh"), data_frame.columns))

    # Add potentially missing columns
    data_frame = _df_add_missing_columns(data_frame)

    # Look up the article type
    data_frame['article_type'] = data_frame['article_type'].map(article_type_lookup, na_action='ignore')

    if clinical_cases_only:
        data_frame = _apply_clinical_case_only(data_frame)

    # Ensure the dataframe can be hashed (i.e., ensure pandas.DataFrame.drop_duplicates does not fail).
    data_frame = _df_make_hashable(data_frame)

    # Obtain a list of disease names
    list_of_diseases = DiseaseOntInterface(cache_path=cache_path, verbose=verbose).pull()['name'].tolist()

    # Run Feature Extracting Tool and Join with `data_frame`.
    if verbose:
        print("\n\nExtracting Features from Text...\n")
    extract = data_frame.progress_apply(lambda x: feature_extract(x, list_of_diseases), axis=1).tolist()
    data_frame = data_frame.join(pd.DataFrame(extract), how='left')

    if verbose:
        print("\n\nCleaning Text Information...\n")
    # Clean the abstract
    data_frame = _df_clean(data_frame)

    return data_frame

















