"""

    Support Tools used Across the BioVida API
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
# Imports
import os
import re
import numpy as np
import pandas as pd
from itertools import chain
from collections import Hashable
from six.moves.html_parser import HTMLParser

# Pull out the unescape function
unescape = HTMLParser().unescape


def dict_reverse(d):
    """

    Reverse a dict.

    :param d:
    :return:
    """
    return {v: k for k, v in d.items()}


def is_int(i):
    """

    Checks if the input (``i``) is an intiger in a way which is robust against booleans.

    :param i: any input
    :type i: ``any``
    :return: whether or not the input is an intiger.
    :rtype: ``bool``
    """
    # Note: ``isinstance(fuzzy_threshold, bool)`` blocks `False` being evaluated as 0 (an intiger).
    return isinstance(i, int) and not isinstance(i, bool)


def multi_replace(s, to_replace):
    """

    Run a replace against ``s`` for every item in ``to_replace``.

    :param s: any string
    :type s: ``str``
    :param to_replace: strings to replace in ``s``
    :type to_replace: ``tuple`` or ``list``
    :return: ``s`` with all items in ``to_replace`` replaced with "".
    :type: ``str``
    """
    for tr in to_replace:
        s = s.replace(tr, "")
    return s


def remove_line_breaks(s):
    """

    Removes all line breaks.

    :param s: any string
    :type s: ``str``
    :return: ``s`` without line breaks.
    :rtype: ``str``
    """
    return re.sub(r'[\t\r\n]', ' ', s)


def remove_html_bullet_points(html):
    """

    Cleans ``html`` by removing any bullet points (html entities)
    and replacing them with semi-colons.

    :param html: html as text with bullet points (HTML entity: '&bull;'
    :type html: ``str``
    :return: see description.
    :rtype: ``str``
    """
    no_points = cln(html).replace('\n&bull; ', "; ").replace('&bull; ', "; ")
    return cln(remove_line_breaks(no_points).replace(" ;", "; "))


def remove_from_head_tail(s, char):
    """

    Remove ``char`` from the head and tail of ``s``.

    :param s: ``s`` as evolved inside ``_abstract_parser()``.
    :type s: ``str``
    :param char: the character to remove from the head and tail of ``s``.
    :type char: ``str``
    :return: see description.
    :rtype: ``str``
    """
    cleaned = cln(s)
    if cleaned.startswith(char):
        cleaned = cleaned[1:]
    if cleaned.endswith(char):
        cleaned = cleaned[:-1]
    return cln(cleaned)


def n_sub_dirs(dir):
    """

    :param dir: a path
    :type dir: ``str``
    :return: number of subdirectories in the dir
    """
    if not os.path.isdir(dir):
        raise FileNotFoundError("'{0}' is not a directory.".format(dir))
    return len([k for i, j, k in os.walk(dir)]) - 1


def pstr(s):
    """

    Convert to any obect to a string using pandas.
    Source: https://github.com/TariqAHassan/EasyMoney

    :param s: item to be converted to a string.
    :type s: ``any``
    :return: a string
    :rtype: ``str``
    """
    return pd.Series([s]).astype('unicode')[0]


def items_null(element):
    """

    Check if an object is a NaN, including all the elements in an iterable.
    Source: https://github.com/TariqAHassan/EasyMoney

    :param element: a python object.
    :type element: ``any``
    :return: assessment of whether or not `element` is a NaN.
    :rtype: ``bool``
    """
    if isinstance(element, (list, tuple)) or 'ndarray' in str(type(element)):
        return True if all(pd.isnull(i) for i in element) else False
    else:
        return pd.isnull(element)


def list_to_bulletpoints(l, sort_elements=True):
    """

    Convert a list to bullet points.

    :param l: a list (in the colloquial sense) of strings.
    :type l: ``list`` or ``tuple``
    :param sort_elements: if ``True``, sort the elements in the list. Defaults to ``True``.
    :type sort_elements: ``bool``
    :return: list itmes formatted as a string of bullet points (with line breaks).
    :rtype: ``str``
    """
    to_format = sorted(l) if sort_elements else list(l)
    return "".join(map(lambda x: "  - '{0}'\n".format(x), to_format))[:-1]


def header(string, flank=True):
    """

    Generate a Header String

    :param string: a string.
    :type string: ``str``
    :param flank: if True, flank the header with line breaks.
    :type flank: ``bool``
    :return:
    """
    # Compute seperating line
    sep_line = "-" * len(string)

    # Display
    if flank:
        print("\n")
    print("\n{0}\n{1}\n{2}\n".format(sep_line, string, sep_line))
    if flank:
        print("\n")


def camel_to_snake_case(name):
    """

    Source: http://stackoverflow.com/a/1176023/4898004

    :param name:
    :return:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def cln(i, extent=1, strip=True):
    """

    String white space 'cleaner'.

    :param i: input str
    :type i: ``str``
    :param extent: 1 --> all white space reduced to length 1; 2 --> removal of all white space.
    :param strip: call str.strip()
    :tyoe strip: ``bool``
    :return: cleaned string
    :rtype: ``str``
    """
    to_return = ""
    if isinstance(i, str) and i != "":
        if extent == 1:
            to_return = re.sub(r"\s\s+", " ", i)
        elif extent == 2:
            to_return = re.sub(r"\s+", "", i)
    else:
        return i

    return to_return.strip() if strip else to_return


def n_split(s, n=2, delim='_'):
    """

    Splits a string into ``n`` groups on a given delimiter.

    Source: http://stackoverflow.com/a/17060409/4898004

    :param s: any string
    :type s: ``str``
    :param n: number of groups. Defaults to `2`.
    :type n: ``int``
    :param delim: a delimiter. Defaults to '_'.
    :type delim: ``str``
    :return: a tuple of with length``n``.
    :rtype: ``tuple``
    """
    groups = s.split(delim)
    return delim.join(groups[:n]), delim.join(groups[n:])


def combine_dicts(dict_a, dict_b):
    """

    :param dict_a:
    :param dict_b:
    :return:
    """
    new = dict_a.copy()
    new.update(dict_b)
    return new


def images_in_dir(dir, return_len=False):
    """

    :param dir:
    :param return_len:
    :return:
    :rtype: ``int`` or ``list``
    """
    if not os.path.isdir(dir):
        raise FileNotFoundError("'{0}' does not exist.".format(str(dir)))
    image_types = (".png", ".jpg", ".tiff", ".gif")
    unnested_list = chain(*[k for i, j, k in os.walk(dir)])
    n_images = [i for i in unnested_list if any(t in i.lower() for t in image_types)]

    return len(n_images) if return_len else n_images


def only_numeric(s):
    """

    Remove all non-numeric characters from a string
    (excluding decimals).

    :param s: a string containing numbers
    :type s: ``str``
    :return: the number contained within ``s``.
    :rtype: ``float`` or ``None``
    """
    # See: http://stackoverflow.com/a/947789/4898004
    cleaned = re.sub(r'[^\d.]+', '', s).strip()
    return float(cleaned) if len(cleaned) else np.NaN


def multimap(data_frame, columns, func):
    """

    Maps some function (``func``) over all columns in ``columns``.

    :param data_frame: as evolved inside ``records_db_merge``.
    :type data_frame: ``Pandas DataFrame``
    :param columns: a 'list' of columns.
    :type columns: ``list`` or ``tuple``
    :param func: some function.
    :type func: ``function``
    :return: ``data_frame`` with ``func`` mapped over all ``columns``.
    :rtype: ``Pandas DataFrame``
    """
    if not isinstance(columns, (list, tuple)):
        return data_frame

    for c in columns:
        data_frame[c] = data_frame[c].map(func, na_action='ignore')

    return data_frame










