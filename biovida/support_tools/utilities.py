"""

    Utilities
    ~~~~~~~~~

"""
# Imports
import os
import shutil
from math import ceil
from pprint import pprint
from collections import defaultdict


class InsufficientNumberOfFiles(Exception):
    pass


def _list_files(path, full=False):
    """

    :param path:
    :param full:
    :return:
    """
    return [os.path.join(path, i) if full else i for i in os.listdir(path) if not i.startswith('.')]


def _categories_in_dir(path, to_block):
    """

    :param path:
    :param to_block:
    :return:
    """
    to_return = list()
    for i in os.listdir(path):
        if os.path.isdir(os.path.join(path, i)) and i not in to_block:
            to_return.append(os.path.join(path, i))
    return to_return


def _list_divide(l, tvt):
    """

    :param l:
    :param tvt:
    :return:
    """
    left, divided_dict = 0, dict()
    for e, (k, v) in enumerate(tvt.items()):
        right = len(l) * v
        # Be greedy if on the last key
        right_rounded = ceil(right) if e == len(tvt.keys()) else int(right)
        divided_dict[k] = l[int(left):int(right_rounded)]
        left += right
    return divided_dict


def train_val_test(data_dir, train=0.8, validation=0.2, test=None, target_dir=None, delete_source=False):
    """

    :param data_dir:
    :param train:
    :param validation:
    :param test:
    :param target_dir:
    :param delete_source:
    :return:
    """
    groups = ('train', 'validation', 'test')
    target_path = data_dir if not isinstance(target_dir, str) else target_dir

    tvt = {k: v for k, v in locals().items() if k in groups and isinstance(v, (float, int)) and not isinstance(v, bool)}
    if sum(tvt.values()) != 1:
        raise ValueError("The following parameters do not sum to 100: {0}.".format(", ".join(sorted(tvt.keys()))))

    existing_dirs = _categories_in_dir(data_dir, to_block=groups)

    existing_files = dict()
    for d in _list_files(data_dir):
        if d not in groups:
            full_path = os.path.join(data_dir, d)
            existing_files[d] = _list_files(full_path, full=True)

    for k, v in existing_files.items():
        for k2, v2 in _list_divide(v, tvt).items():
            target = os.path.join(target_path, os.path.join(k2, k))
            if not os.path.isdir(target):
                os.makedirs(target)
            for i in v2:
                new_location = os.path.join(target, os.path.basename(i))
                shutil.copy2(i, new_location)

    if delete_source:
        for i in existing_dirs:
            shutil.rmtree(i)













































