"""

    Cancer Imaging Archive Parameters Extraction
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import re
import requests
import pandas as pd
from itertools import chain

from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import n_split
from biovida.support_tools.support_tools import combine_dicts
from biovida.support_tools.support_tools import camel_to_snake_case


def _roll_strs_forward(l):
    """

    Replaces gaps between strings in an list with
    the last string to appear in the list.

    :param l: a python list.
    :type l: ``list``
    :return: ``l`` with strings in the list rolled forward (i.e., replacing) all non-strings.
    :rtype: ``list``
    """
    current_string = None
    for i in range(len(l)):
        if isinstance(l[i], str):
            current_string = cln(l[i])
        elif current_string is not None:
            l[i] = current_string
    return l


def _contains_all_cols(column_names):
    """

    Test to see if the header contains all of the required column names.

    :param column_names: column names of the dataframe.
    :type column_names: ``Pandas Series``
    :return:
    """
    required_colnames = ("resource", "queryendpoint", "query parameters", "format", "description")
    return all(any(rc in h for rc in required_colnames) for h in column_names.str.lower())


def _extract_on_required(q_params):
    """

    Parse Strings which contain '(R)'.

    :param q_params: a string containing '(R)'.
    :type q_params: ``str``
    :return: parsed string
    :rtype: ``list``

    :Example:
    >>> _extract_on_required('Date (R)Collection(R)PatientID')
    ...
    ['Date (R)', 'Collection (R)', 'PatientID']
    """
    req_split = re.split('(\(R\))', cln(q_params))

    final = list()
    for i in range(len(req_split)):
        if i != len(req_split) - 1 and req_split[i + 1] == "(R)":
            final.append([req_split[i] + req_split[i + 1]])
        elif req_split[i] != '(R)':
            final.append(req_split[i].split())

    return list(chain(*final))


def _query_parameters_parser(q_params):
    """

    Parse rows in query_parameters column into a list of tuples.

    :param q_params: query_parameters
    :type q_params: ``str``
    :return: a list of tuples of the form: ``[('PARAM_1', 'r' or 'o'), 'PARAM_2', 'r' or 'o'),...]``,
            where 'r' denotes a required parameter and 'o' denotes an optional parameter.
    :rtype: ``list``
    """
    if q_params.strip().lower() == 'none':
        return None
    elif "(R)" in q_params.strip():
        parsed_q_params = _extract_on_required(q_params)
    else:
        parsed_q_params = list(map(lambda i: i.strip(), cln(q_params).split("/")))

    # Mark params as required
    return [(p.replace("(R)", "").strip(), 'r') if "(R)" in p else (p, 'o') for p in parsed_q_params]


def _trcia_api_table_from_html(table_loc):
    """

    Extract the TCIA API Reference Table from the Usage Guide Wiki.

    :param table_loc: url to the TCIA API Usage Guide.
    :type table_loc: ``str``
    :return: the TCIA API Reference Table with unrefined column.
    :rtype: ``Pandas DataFrame``
    """
    html = requests.get(table_loc).text

    # Extract all tables from the page
    all_tables = pd.read_html(str(html), header=0)

    # Keep only those tables with valid headers
    valid_tables = [t for t in all_tables if _contains_all_cols(t.columns)]

    if len(valid_tables) != 1:
        raise AttributeError("Multiple Valid API Reference Tables Found.")

    # Extract the api reference table
    api_df = valid_tables[0]

    # Clean Column Names
    c_names = list(map(lambda i: camel_to_snake_case(cln(i, extent=2)), api_df.columns))
    api_df.columns = list(map(lambda i: n_split(i, n=2)[0].strip(), c_names))

    return api_df


def reference_table(table_loc='https://wiki.cancerimagingarchive.net/display/Public/'
                             'TCIA+Programmatic+Interface+%28REST+API%29+Usage+Guide'):
    """

    Extract and Parse the API Reference Table from the Cancer Image Archive Usage Guide Wiki.

    :param table_loc: URL to the TCIA API Usage Guide.
    :type table_loc: ``str``
    :return: TCIA API Reference Table.
    :rtype: ``Pandas DataFrame``
    """
    # Extract the c
    api_df = _trcia_api_table_from_html(table_loc)

    # Roll the strings in 'Resource' forward.
    api_df['resource'] = _roll_strs_forward(api_df['resource'].tolist())

    # Clean all columns
    for c in api_df.columns:
        api_df[c] = [cln(i) if isinstance(i, str) else i for i in api_df[c].tolist()]

    # Remove Rows with no 'query_endpoint'
    api_df = api_df[pd.notnull(api_df['query_endpoint'])].reset_index(drop=True)

    # Parse the 'format' column
    api_df['format'] = api_df['format'].map(lambda x: x.lower().split("/"), na_action='ignore')

    # Parse the 'query_parameters' column
    api_df['query_parameters'] = api_df['query_parameters'].map(_query_parameters_parser)

    return api_df


def reference_table_as_dict(table=None):
    """

    Return a nested dict of the Cancer Image Archive API Reference table.

    :param table: Cancer Image Archive API Reference table. If ``None``, the table will be extracted from the web
                  directly. Defaults to ``None``.
    :type table: ``Pandas DataFrame`` or None.
    :return: dictionary of the form:

            ``{'resource': {'query_endpoint': {'query_parameters': ..., 'format': ..., 'description': ...}, ..., ...}``

    :rtype: ``dict``
    """
    # Use `table` if it is a DataFrame, otherwise download the table from the web.
    api_df = table if 'DataFrame' in str(type(table)) else reference_table()

    nested_dict = dict()
    for (r, qe, qp, f, d) in zip(*[api_df[c].tolist() for c in api_df.columns]):
        row_data = {qe: {"query_parameters": qp, "format": f, "description": d}}
        if r not in nested_dict:
            nested_dict[r] = row_data
        else:
            nested_dict[r] = combine_dicts(nested_dict[r], row_data)

    return nested_dict




















