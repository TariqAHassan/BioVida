"""

    General Support Tools for Open-i Data Processing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import re
from itertools import chain
from urllib.parse import urlsplit # handle python 2

# General Support tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import items_null
from biovida.support_tools._support_data import ages_as_tuples_rev

# Regex for ``extract_float()``
non_decimal = re.compile(r'[^\d.]+')

def _age_word_regex(age_num_word):
    _mold = "(^| ){0}( |,|\.|\?|!|$)"
    if '-' not in age_num_word:
        to_compile = _mold.format(age_num_word)
    else:
        to_compile = _mold.format(age_num_word.replace("-", "[-| ]"))
    return re.compile(to_compile, flags=re.I)

_age_match = tuple([(" {0} ".format(n), _age_word_regex(w)) for w, n in ages_as_tuples_rev])


# ----------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------


nonessential_openi_columns = [
    'biovida_version',
    'detailed_query_url',
    'fulltext_html_url',
    'get_article_figures',
    'img_grid150',
    'img_large',
    'img_thumb',
    'img_thumb_large',
    'pmc_url',
    'pub_med_url',
    'license_url',
    'medpix_article_id',
    'medpix_figure_id',
    'medpix_image_url',
    'similar_in_collection',
    'similar_in_results'
]


possible_openi_image_processing_cols = [
    'grayscale',
    'medpix_logo_bounding_box',
    'hbar',
    'hborder',
    'vborder',
    'upper_crop',
    'lower_crop',
    'visual_image_problems',
    'valid_image'
]


# ----------------------------------------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------------------------------------


class ImageProblemBasedOnText(Exception):
    pass


def iter_join(t, join_on="_"):
    """

    Convert an iterabe (``t``) to a string by joining on ``join_on``.

    :param t: any iterable.
    :type t: ``iterable``
    :param join_on: a string to join ``t`` on.
    :type join_on: ``str``
    :return: ``t`` as a string.
    :rtype: ``str``
    """
    return join_on.join(t) if isinstance(t, (list, tuple)) else t


def null_convert(i):
    """

    Convert ``i`` to ``None`` if it is void else ``i`` is returned.

    :param i: any.
    :type i: ``any``
    :return: None ``i`` is void (e.g., '' or []) else ``i``.
    :rtype: ``None`` else ``type(i)``
    """
    return None if not i else i


def url_combine(url1, url2):
    """

    Combines two urls into one.

    :param url1: a URL.
    :type url1: ``str``
    :param url2: a URL.
    :type url2: ``str``
    :return: a string of the form ``url1/url2``.
    :rtype: ``None`` or ``str``
    """
    if any(x is None or items_null(x) for x in [url1, url2]):
        return None
    else:
        left = url1[:-1] if url1.endswith("/") else url1
        right = url2[1:] if url2.startswith("/") else url2
        return "{0}/{1}".format(left, right)


def item_extract(i, list_len=1):
    """

    Extract the first item in a list or tuple, else return ``None``.

    :param i: any list or tuple.
    :type i: ``list`` or ``tuple``
    :param list_len: required length of ``i``. Defaults to `1`.
    :type list_len: ``int``
    :return: the first element in ``i`` if ``i`` is a ``list`` or ``tuple`` and ``len(i) == list_len``.
    :rtype: ``type(i[0])`` or ``None``
    """
    return i[0] if isinstance(i, (list, tuple)) and len(i) == list_len else None


def extract_float(i):
    """

    Extract a d from a string ``i``

    :param i: a string.
    :type i: ``str``
    :return: the floating point number in ``i`` as a string.
    :rtype: ``str``
    """
    # Source: http://stackoverflow.com/a/947789/4898004
    return non_decimal.sub('', i)


def multiple_decimal_remove(s):
    """

    Remove multiple decimal from a string (``s``)

    :param s: a string.
    :type s: ``str``
    :return: ``s`` with only one.
    :rtype: ``None`` or ``str``
    """
    if isinstance(s, (int, float)):
        return s

    s_cln = cln(extract_float(str(s)), extent=2)
    if s_cln.startswith('.') and s_cln.endswith('.'):
        return None

    s_repeats_rmv = re.sub(r'\.+', '.', s_cln)
    if s_repeats_rmv.count('.') > 1:
        return None
    else:
        return cln(s_repeats_rmv, extent=2)


def filter_unnest(l, filter_for=None):
    """

    Filters out ``filter_for`` and flattens ``l``.

    :param l: a 2D iterable.
    :type l: ``iterable``
    :param filter_for: items to be removed. Defaults to ``None``.
    :type filter_for: any
    :return: ``l`` flattened with ``filter_for`` removed.
    :rtype: ``list``
    """
    return list(chain(*filter(filter_for, l)))


def num_word_to_int(input_str):
    """

    Replace natural numbers from 1 to 105 with integers.

    :param input_str: any string.
    :rtype input_str: ``str``
    :return: ``input_str`` with all numbers from 1-130 in natural language (e.g., 'twenty-five')
             with the integer equivalent.
    :rtype: ``str``

    :Example:

    >>> num_word_to_int(input_str="the disease was present from twenty  five until their death at eighty-seven.")
    ...
    'the disease was present from 25 until their death at 87'
    ...
    # However, the simple presence of the substring 'one' doesn't confuse the procedure.
    >>> num_word_to_int(input_str="the oneway (sic) street is closed due to construction.")
    ...
    'the oneway (sic) street is closed due to construction'
    ...
    # It can also handle common punctuation (though it won't conserve it -- not currently needed however).
    >>> num_word_to_int(input_str="The ring was the only one, in all of history, to xyz")
    ...
    'The ring was the only 1 in all of history, to xyz'
    ...
    # It is also not confused by common expressions (assuming the number is < 20).
    >>> num_word_to_int(input_str="one-shot learning is the ultimate goal of AI research.")
    'Robust one-shot learning is the one of the goals of AI research.'
    """
    # ToDo: faster solutions are surely possible...Without this step,
    #       biovida.images._interface_support.openi._openi_text_feature_extraction is ~4x faster.
    # Note: the reversal (ages_as_tuples_rev) ensures that 'twenty five' yields '25', not '20 5'.
    input_str_clean = " ".join(input_str.split()).strip()
    for (n, w) in _age_match:
        input_str_clean = re.sub(w, n, input_str_clean)
    return " ".join(input_str_clean.split()).strip()


def url_path_extract(url):
    """

    Extracts the path for a given URL.

    :param url: a Uniform Resource Locator (URL).
    :rtype url: ``str``
    :return: the path for ``url``.
    :rtype: ``str``
    """
    return urlsplit(url).path[1:].replace("/", "__")











