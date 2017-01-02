"""

    Support Tools for the `openi_interface.py` module
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
# Imports
import inflect
from itertools import chain


# Other Tools
p = inflect.engine()
non_decimal = re.compile(r'[^\d.]+')
age_dict = {p.number_to_words(i): i for i in range(1, 135)}


def openi_bounds_formatter(bounds):
    """Format the computed bounds for the Open-i API."""
    return ["&m={0}&n={1}".format(i[0], i[1]) for i in bounds]

def iter_join(t, join_on="_"):
    return join_on.join(t) if isinstance(t, (list, tuple)) else i

def null_convert(i):
    return None if not i else i

def url_combine(url1, url2):
    if any(x is None for x in [url1, url2]):
        return None
    else:
        return (url1[:-1] if url1.endswith("/") else url1) + url2

def item_extract(i, list_len=1):
    """Extract the first item in a list or tuple, else return None"""
    return i[0] if isinstance(i, (list, tuple)) and len(i) == list_len else None

def extract_float(i):
    """see: http://stackoverflow.com/a/947789/4898004"""
    return non_decimal.sub('', i)

def filter_unnest(l, filter_for=None):
    return list(chain(*filter(filter_for, l)))

def num_word_to_int(input_str):
    """Replace natural numbers from 1 to 130 with intigers."""
    for w, i in age_dict.items():
        for case in [w.upper(), w.lower(), w.title()]: # not perfect, but should do
            input_str = input_str.replace(case, str(i))
    return input_str

