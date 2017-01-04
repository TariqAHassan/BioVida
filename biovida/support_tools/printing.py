"""

    Printing Tools
    ~~~~~~~~~~~~~~


"""
# Imports
import pandas as pd

# General Support Tools
from biovida.support_tools.support_tools import pstr
from biovida.support_tools.support_tools import items_null


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
    if full_rows: pd.set_option('display.max_rows', len(pd_df))
    if full_cols: pd.set_option('expand_frame_repr', False)

    # Print the data frame
    print(pd_df)

    if full_rows: pd.reset_option('display.max_rows')
    if full_cols: pd.set_option('expand_frame_repr', True)


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























