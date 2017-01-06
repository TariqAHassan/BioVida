"""

    Support Tools for the `images` subpackage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
# Imports
import re
import inflect
from itertools import chain
from urllib.parse import urlsplit # handle python 2

# General Support tools
from biovida.support_tools.support_tools import cln

# Other Tools
p = inflect.engine()
non_decimal = re.compile(r'[^\d.]+')
age_dict = {p.number_to_words(i): i for i in range(1, 135)}


def iter_join(t, join_on="_"):
    """

    :param t:
    :param join_on:
    :return:
    """
    return join_on.join(t) if isinstance(t, (list, tuple)) else i

def null_convert(i):
    """

    :param i:
    :return:
    """
    return None if not i else i

def url_combine(url1, url2):
    """

    :param url1:
    :param url2:
    :return:
    """
    if any(x is None for x in [url1, url2]):
        return None
    else:
        return (url1[:-1] if url1.endswith("/") else url1) + url2

def item_extract(i, list_len=1):
    """

    Extract the first item in a list or tuple, else return None

    :param i:
    :param list_len:
    :return:
    """
    return i[0] if isinstance(i, (list, tuple)) and len(i) == list_len else None

def extract_float(i):
    """

    :param i:
    :return:
    """
    # Source: http://stackoverflow.com/a/947789/4898004
    return non_decimal.sub('', i)

def multiple_decimal_remove(s):
    """

    :param s: a string
    :type s: ``str``
    :return:
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

def numb_extract(string, join_on=""):
    """

    :param string:
    :param join_on:
    :return:
    """
    return join_on.join(re.findall(r'[0-9]+', string))

def filter_unnest(l, filter_for=None):
    """

    :param l:
    :param filter_for:
    :return:
    """
    return list(chain(*filter(filter_for, l)))

def num_word_to_int(input_str):
    """

    Replace natural numbers from 1 to 130 with intigers.

    :param input_str:
    :return:
    """
    for w, i in age_dict.items():
        for case in [w.upper(), w.lower(), w.title()]: # not perfect, but should do
            input_str = input_str.replace(case, str(i))
    return input_str

def url_path_extract(url):
    """

    :param url:
    :return:
    """
    return urlsplit(url).path[1:].replace("/", "__")



























