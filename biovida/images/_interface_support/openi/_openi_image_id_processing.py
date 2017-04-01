# coding: utf-8

"""

    Processing the 'image_id' Column Returned by Open-i
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import re
from collections import defaultdict

# General Support Tools
from biovida.support_tools.support_tools import cln, only_numeric, multi_replace


_image_id_cleaner_regex = re.compile('\.[A-Za-z]+$')
_image_id_medpix_regex = re.compile('\d+$')
_image_id_trail_regex = re.compile('\d+-\d+(?:[a-z])?$')
_image_id_zero_regex = re.compile('^0+')


def _image_id_parser(image_id):
    """

    Extract the following pattern from ``image_id``:
    'f...numbers(optional: one letter from a-g)'.

    :param image_id: an element from the 'image_id' column.
    :type image_id: ``str``
    :return: see description and examples in ``_image_id_short()``.
    :rtype: ``str`` or ``None``
    """
    allowed_letters = list('abcdefghij')
    if 'f' in image_id:
        look_for = 'f'
    elif 'g' in image_id:
        look_for = 'g'
    else:
        return None

    candidate = ''
    do_add = False
    hit_digit = False
    letters_added = False
    drop_letters = False
    for c in image_id.lower():
        if c == look_for:
            do_add = True
            continue
        if do_add and c.isdigit():
            candidate += c
            hit_digit = True
        elif hit_digit and not letters_added and c in allowed_letters:
            candidate += c
            letters_added = True
            continue
        if hit_digit and letters_added and c.isalpha():
            drop_letters = True
        if hit_digit and not c.isdigit():
            break
    if len(candidate) and drop_letters:
        return only_numeric(candidate, float_output=False)
    elif len(candidate):
        return candidate
    else:
        return None


def _image_id_cleaner(image_id):
    """

    Performs the following changes
    to ``image_id``.

    1. File extensions
    2. Convert 'image' and 'img' to 'f'
    3. Clean white spaces and lower.

    :param image_id: an element from the 'image_id' column.
    :type image_id: ``str``
    :return: see description.
    :rtype: ``str``
    """
    image_id_no_ext = re.sub(_image_id_cleaner_regex, "", image_id)
    return cln(multi_replace(image_id_no_ext, ('image', 'img'), replace_with='f')).lower()


def _image_id_short(journal_title, image_id):
    """

    Simplify ``image_id`` to determine the figure number.

    :param journal_title: an element from the 'journal_title' column.
    :type journal_title: ``None`` or ``str``
    :param image_id: an element from the 'image_id' column.
    :type image_id: ``str``
    :return: a cleaned ``image_id``
    :rtype: ``str`` or ``None``

    :Example:

    >>> _image_id_short(journal_title='medpix', image_id='MPX..16962')
    ...
    '16962'
    >>> _image_id_short(journal_title='pubmed', image_id='f06')
    ...
    '6'
    >>> _image_id_short(journal_title='pubmed', image_id='f9')
    ...
    '9'
    >>> _image_id_short(journal_title='pubmed', image_id='new-img009-mri')
    ...
    '9'
    >>> _image_id_short(journal_title='pubmed', image_id='image0011')
    ...
    '11'
    >>> _image_id_short(journal_title='unknown', image_id='figure4a')
    ...
    '4a'
    >>> _image_id_short(journal_title='unknown', image_id='figure4az')
    ...
    '4'
    >>> _image_id_short(journal_title='pubmed', image_id='f1b-new')
    ...
    '1b'
    >>> _image_id_short(journal_title=None, image_id='f1bz')
    ...
    '1'
    >>> _image_id_short(journal_title='pubmed', image_id='f019c-2')
    ...
    '2'
    >>> _image_id_short(journal_title='pubmed', image_id='f319cg-33')
    ...
    '33'
    >>> _image_id_short(journal_title=np.NaN, image_id='f2b-993')
    ...
    '2b' # not confused by the trailing '999' as it is >> 2, and thus less likely to be the fig. no.
    """
    if not isinstance(image_id, str):
        return None

    image_id_clean = _image_id_cleaner(image_id)

    if isinstance(journal_title, str) and 'medpix' in journal_title.lower():
        medpix_match = re.findall(_image_id_medpix_regex, image_id_clean)
        if len(medpix_match) == 1 and len(medpix_match[0]):
            return medpix_match[0]

    end = re.findall(_image_id_trail_regex, image_id_clean)
    end_match = end[0].replace("-", "") if len(end) == 1 and len(end[0]) else None
    p_image_id = _image_id_parser(image_id_clean)

    def output_decision(a, b):
        small_ceiling = 100
        to_compare = list()
        for i in (a, b):
            try:
                to_compare.append([i, only_numeric(i)])
            except:
                pass
        if len(to_compare) != 2:
            return None
        if any(i[1] < small_ceiling for i in to_compare):
            return min(to_compare, key=lambda x: x[1])[0]
        else:
            return max(to_compare, key=lambda x: x[1])[0]

    def output_clean(output):
        if len([i for i in output if i.isdigit() and i != '0']):
            return cln(re.sub(_image_id_zero_regex, '', output))
        else:
            return cln(output)

    if end_match is None and p_image_id is None:
        return None
    elif isinstance(end_match, str) and isinstance(p_image_id, str):
        fig_no = output_decision(end_match, p_image_id)
    elif isinstance(end_match, str):
        fig_no = end_match
    elif isinstance(p_image_id, str):
        fig_no = p_image_id
    return output_clean(fig_no)


def _image_id_short_enforce_unique(data_frame):
    """

    If the 'image_id_short' column is not unique
    with respect to 'uid', make it unique by adding a
    an integer.

    :param data_frame: as evolved inside ``image_id_short_gen()``
    :type data_frame: ``Pandas DataFrame``
    :return: see description
    :rtype: ``Pandas DataFrame``


    >>> print(data_frame)
    ...
                 uid             image_id        image_id_short
        1    PMC4372763  sap-27-01-052-g002.tif         None
        2   PMC4593917              FI0192cr-1          192
        3   PMC4593917              FI0192cr-2          192
    ...
    >>> print(_image_id_short_enforce_unique(data_frame))
    ...
                  uid             image_id        image_id_short
        1    PMC4372763  sap-27-01-052-g002.tif           1
        2   PMC4593917              FI0192cr-1        192_1
        3   PMC4593917              FI0192cr-2        192_2

    """
    def not_unique_image_id_short(row):
        """Check if elements in the 'image_id_short' series are unique."""
        image_id_short = row['image_id_short'].tolist()
        if any(not isinstance(i, str) for i in image_id_short):
            return True
        return not len(set(image_id_short)) == len(image_id_short)

    # Dict of the form: {'uid': not unique or unique (boolean), ...}
    invalid_si_id_dict = data_frame.groupby('uid').apply(not_unique_image_id_short).to_dict()

    d = defaultdict(int)
    # Iterate over only those rows where 'image_id_short' is not unique and correct accordingly.
    for index, row in data_frame.iterrows():
        if invalid_si_id_dict.get(row['uid']):
            d[row['uid']] += 1
            if isinstance(row['image_id_short'], str):
                new_name = "{0}_{1}".format(row['image_id_short'],str(d[row['uid']]))
                data_frame.set_value(index, 'image_id_short', new_name)
            else:
                data_frame.set_value(index, 'image_id_short', str(d[row['uid']]))
    return data_frame


def image_id_short_gen(data_frame):
    """

    Simplify 'image_id' by 'boiling it down'
    to the figure number.

    Note: this will mutate ``data_frame`` in memory.

    :param data_frame: an Openi ``records_db`` or ``cache_records_db``.
    :type data_frame: ``Pandas DataFrame``
    :return: ``data_frame`` with a ``image_id_short`` column.
    :rtype: ``Pandas DataFrame``
    """
    data_frame['image_id_short'] = data_frame.apply(
        lambda x: _image_id_short(x['journal_title'], x['image_id']), axis=1)

    return _image_id_short_enforce_unique(data_frame)
