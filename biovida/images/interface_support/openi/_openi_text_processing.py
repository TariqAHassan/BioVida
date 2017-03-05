"""

    Tools to Clean Raw Open-i Text
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import numpy as np
import pandas as pd

# General Support tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import unescape

# Image Support Tools
from biovida.images._image_tools import resetting_label

# General Support Tools
from biovida.support_tools.support_tools import camel_to_snake_case

# Open-i API Parameters Information
from biovida.images.interface_support.openi._openi_parameters import openi_image_type_params
from biovida.images.interface_support.openi._openi_parameters import openi_article_type_params
from biovida.images.interface_support.openi._openi_parameters import openi_image_type_modality_full

# Tools for Text Feature Extraction
from biovida.images.interface_support.openi.openi_text_feature_extraction import feature_extract
from biovida.images.interface_support.openi.openi_text_feature_extraction import _html_text_clean

# Other BioVida APIs
from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface


def _mesh_cleaner(mesh):
    """

    Clean mesh terms by cleaning them.

    :param mesh: a list of mesh terms.
    :type mesh: ``tuple`` or ``list``
    :return: a cleaned tuple of mesh terms.
    :rtype: ``tuple`` or ``type(mesh)``
    """
    if isinstance(mesh, (list, tuple)):
        return tuple([cln(unescape(i)) if isinstance(i, str) else i for i in mesh])
    else:
        return mesh


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
    for c in ('mesh_major', 'mesh_minor'):
        data_frame[c] = data_frame[c].map(_mesh_cleaner, na_action='ignore')

    # Convert all elements in the 'license_type' and 'license_url' columns to string.
    for c in ('license_type', 'license_url'):
        data_frame[c] = data_frame[c].map(lambda x: "; ".join(map(str, x)) if isinstance(x, (list, tuple)) else x,
                                          na_action='ignore')

    return data_frame


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

    return data_frame


def _df_clean(data_frame):
    """

    Clean the text information.

    :param data_frame: the dataframe evolved inside ``openi_raw_extract_and_clean``.
    :type data_frame: ``Pandas DataFrame``
    :return: cleaned ``data_frame``.
    :rtype:  ``Pandas DataFrame``
    """
    # Clean the abstract
    data_frame['abstract'] = data_frame.apply(
        lambda x: _html_text_clean(x['abstract'], 'both', parse_medpix='medpix' in str(x['journal_title']).lower()),
        axis=1)

    # Add the full name for modalities (before the 'image_modality_major' values are altered below).
    data_frame['modality_full'] = data_frame['image_modality_major'].map(
        lambda x: openi_image_type_modality_full.get(cln(x).lower(), x), na_action='ignore')

    # Make the type of Imaging technology type human-readable. ToDo: apply to the other image_modality.
    data_frame['image_modality_major'] = data_frame['image_modality_major'].map(
        lambda x: openi_image_type_params.get(cln(x).lower(), x), na_action='ignore')

    # Look up the article type
    data_frame['article_type'] = data_frame['article_type'].map(
        lambda x: openi_article_type_params.get(cln(x).lower(), x), na_action='ignore')

    # Label the number of instance of repeating 'uid's.
    data_frame['uid_instance'] = resetting_label(data_frame['uid'].tolist())

    # Replace missing Values with with NaN and Return
    return _df_fill_nan(data_frame)


def openi_raw_extract_and_clean(data_frame, verbose, cache_path):
    """

    Extract features from, and clean text of, ``data_frame``.

    :param data_frame: the dataframe evolved inside ``biovida.images.openi_interface._OpeniRecords().records_pull()``.
    :rtype data_frame: ``Pandas DataFrame``
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

    # Ensure the dataframe can be hashed (i.e., ensure pandas.DataFrame.drop_duplicates does not fail).
    data_frame = _df_make_hashable(data_frame)

    # Obtain a list of disease names
    list_of_diseases = DiseaseOntInterface(cache_path=cache_path, verbose=verbose).pull()['name'].tolist()

    # Run Feature Extracting Tool and Join with `data_frame`.
    if verbose:
        print("\n\nExtracting Text Features...\n")
    pp = pd.DataFrame(data_frame.progress_apply(
        lambda x: feature_extract(x, list_of_diseases=list_of_diseases), axis=1).tolist()).fillna(np.NaN)
    data_frame = data_frame.join(pp, how='left')

    if verbose:
        print("\n\nCleaning Text Information...\n")
    # Clean the abstract
    data_frame = _df_clean(data_frame)

    return data_frame



















