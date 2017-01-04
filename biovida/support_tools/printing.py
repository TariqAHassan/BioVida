"""

    Printing Tools
    ~~~~~~~~~~~~~~


"""
# Imports
import pandas as pd
from copy import deepcopy

# General Support Tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import pstr
from biovida.support_tools.support_tools import items_null


# ---------------------------------------------------------------------------------------------
# Dictionary Printing
# ---------------------------------------------------------------------------------------------


def _key_padding(s, len_longest_key):
    """

    Add padding a key in a dict. for `dict_pretty_printer()`.

    :param s:
    :param len_longest_key:
    :return:
    """
    if not isinstance(s, str):
        return s
    return "{0}:{1}".format(cln(s).replace("_", " ").title(), " " * abs(len_longest_key - len(s)))


def _value_padding(s, len_longest_key, print_mold):
    """

    Add padding a value in a dict. for `dict_pretty_printer()`.

    :param s:
    :param len_longest_key:
    :return:
    """
    if not isinstance(s, str):
        return s
    padding = len_longest_key + (len(print_mold) - 3) + 2  # 2 = 1 space for the colon + 1 space added by `print()`.
    return s.replace("\n", "\n{0}".format(" " * padding))


def _char_in_braces(full_s, char_position):
    """

    Checks if a given position in a string lies between braces.
    Note: This is a rough solution, but more than sufficient. A robust solution would require a state machine.

    :param full_s: a string.
    :type full_s: ``str``
    :param char_position: a position in the string.
    :type char_position: ``int``
    :return: ``True`` if yes, else ``False``
    :rtype: bool
    """
    if not isinstance(full_s, str):
        raise ValueError('`full_s` must be a string.')

    if not isinstance(char_position, int):
        raise ValueError('`char_position` must be an int.')

    for pair in ('()', '[]', '{}'):
        if pair[0] in full_s[:char_position] and pair[1] in full_s[char_position:]:
            return True
    else:
        return False


def _value_correction(s, len_longest_key, max_value_length, print_mold):
    """

    Formats a value in a dict. for `dict_pretty_printer()`.

    :param s:
    :param len_longest_key:
    :param max_value_length:
    :param print_mold:
    :return:
    """
    if not isinstance(s, str):
        return s

    # Clean the input
    s_cleaned = cln(s).replace("\n", " ")

    # Get the position of all spaces in the cleaned string -- spaces inside braces are excluded.
    spaces = [i for i, c in enumerate(s_cleaned) if c == " " and not _char_in_braces(s_cleaned, i)]

    # Return if the string is shorter than the `max_value_length` or there are no space.
    if len(s_cleaned) < max_value_length or not len(spaces):
        return _value_padding(s_cleaned, len_longest_key, print_mold)

    # Identify ideal points for a line break, w.r.t. max_value_length.
    ideal_break_points = [i for i in range(len(s_cleaned)) if i % max_value_length == 0 and i != 0]

    # Get the possible break points
    true_break_points = [min(spaces, key=lambda x: abs(x - ideal)) for ideal in ideal_break_points]

    if len(true_break_points):
        formatted_string = "".join([c if e not in true_break_points else "\n" for e, c in enumerate(s_cleaned)])
    else:
        formatted_string = s_cleaned

    return _value_padding(formatted_string, len_longest_key, print_mold)


def dict_pretty_printer(d, max_value_length=70):
    """

    Pretty prints a dictionary with vertically aligned values.

    Example:
        - Key1:       Value1
        - Key12:      Value12
        - Key123:     Value123
        - Key1234:    Value1234
        - Key12345:   Value12345

    :param d: a dictionary
    :type d: ``dict``
    :param max_value_length: max. number of characters in a string before a line break.
                            This is a fuzzy threshold because the algorithm above will only insert
                            line breaks where there are already spaces and will not insert line breaks
                            between braces.
    :type max_value_length: ``int``
    """
    print_mold = " - {0} "

    # Compute the length of the longest key
    len_longest_key = len(max(list(d.keys()), key=len))

    new_dict = {_key_padding(k, len_longest_key): _value_correction(v, len_longest_key, max_value_length, print_mold)
                for k, v in deepcopy(d).items()}

    # Print the dict
    for k, v in new_dict.items():
        print(print_mold.format(k), v)


# ----------------------------------------------------------------------------------------------------------
# Pandas Printing Suit
# ----------------------------------------------------------------------------------------------------------


def _padding(s, amount, justify):
    """

    Add padding to a string.
    Source: https://github.com/TariqAHassan/EasyMoney

    :param s: a string.
    :type s: ``str``
    :param amount: the amount of white space to add.
    :type amount: ``float`` or ``int``
    :param justify: the justification of the resultant text. Must be one of: 'left' or 'center'.
    :type justify: ``str``
    :return: `s` justified, or as passed if `justify` is not one of: 'left' or 'center'.
    :rtype: ``str``
    """
    pad = ' ' * amount
    if justify == 'left':
        return "%s%s" % (pstr(s), pad)
    elif justify == 'center':
        return "%s%s%s" % (pad[:int(amount/2)], pstr(s), pad[int(amount/2):])
    else:
        return s


def _pandas_series_alignment(pandas_series, justify):
    """

    Align all items in a pandas series.
    Source: https://github.com/TariqAHassan/EasyMoney

    :param pandas_series: a pandas series.
    :type pandas_series: ``Pandas Series``
    :param justify: the justification of the resultant text. Must be one of: 'left', 'right' or 'center'.
    :type justify: ``str``
    :return: aligned series
    :rtype: ``str``
    """
    if justify == 'right':
        return pandas_series
    longest_string = max([len(s) for s in pandas_series.astype('unicode')])
    return [_padding(s, longest_string - len(s), justify) if not items_null(s) else s for s in pandas_series]


def align_pandas(data_frame, to_align='right'):
    """

    Align the columns of a Pandas DataFrame by adding whitespace.
    Source: https://github.com/TariqAHassan/EasyMoney

    :param data_frame: a dataframe.
    :type data_frame: ``Pandas DataFrame``
    :param to_align: 'left', 'right', 'center' or a dictionary of the form: ``{'Column': 'Alignment'}``.
    :type to_align: ``str``
    :return: dataframe with aligned columns.
    :rtype: ``Pandas DataFrame``
    """
    if isinstance(to_align, dict):
        alignment_dict = to_align
    elif to_align.lower() in ['left', 'right', 'center']:
        alignment_dict = dict.fromkeys(data_frame.columns, to_align.lower())
    else:
        raise ValueError("to_align must be either 'left', 'right', 'center', or a dict.")

    for col, justification in alignment_dict.items():
        data_frame[col] = _pandas_series_alignment(data_frame[col], justification)

    return data_frame


def pandas_print_full(pd_df, full_rows=False, full_cols=True):
    """

    Print *all* of a Pandas DataFrame.
    Source: https://github.com/TariqAHassan/EasyMoney

    :param pd_df: DataFrame to printed in its entirety.
    :type pd_df: ``Pandas DataFrame``
    :param full_rows: print all rows if True. Defaults to False.
    :type full_rows: ``bool``
    :param full_cols: print all columns side-by-side if True. Defaults to True.
    :type full_cols: ``bool``
    """
    if full_rows:
        pd.set_option('display.max_rows', len(pd_df))
    if full_cols:
        pd.set_option('expand_frame_repr', False)

    # Print the data frame
    print(pd_df)

    if full_rows:
        pd.reset_option('display.max_rows')
    if full_cols:
        pd.set_option('expand_frame_repr', True)


def pandas_pretty_print(data_frame, col_align='right', header_align='center', full_rows=True, full_cols=True):
    """

    Pretty Print a Pandas DataFrame.
    Source: https://github.com/TariqAHassan/EasyMoney

    :param data_frame: a dataframe.
    :type data_frame: ``Pandas DataFrame``
    :param col_align: 'left', 'right', 'center'' or a dictionary of the form: ``{'Column': 'Alignment'}``.
    :type col_align: ``str`` or ``dict``
    :param header_align: alignment of headers. Must be one of: 'left', 'right', 'center'.
    :type header_align: ``str`` or ``dict``
    :param full_rows: print all rows.
    :type full_rows: ``bool``
    :param full_cols: print all columns.
    :type full_cols: ``bool``
    """
    aligned_df = align_pandas(data_frame, col_align)
    pd.set_option('colheader_justify', header_align)
    pandas_print_full(aligned_df.fillna(""), full_rows, full_cols)
    pd.set_option('colheader_justify', 'right')






































