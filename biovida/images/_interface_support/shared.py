# coding: utf-8

"""

    Shared Interface Support Tools
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import pandas as pd


def save_records_db(data_frame, path):
    """

    :param data_frame:
    :param path:
    :return:
    """
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("`records_db` is not a DataFrame.")
    elif os.path.isdir(path):
        raise IsADirectoryError("'{0}' is a directory.".format(path))
    save_path = "{0}.p".format(path) if not path.endswith(".p") else path
    data_frame.to_pickle(save_path)
