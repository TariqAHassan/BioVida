"""

    Utilities
    ~~~~~~~~~

"""
# Imports
import os
import shutil
import numpy as np
from math import ceil
from scipy.ndimage import imread

from biovida.support_tools.support_tools import isclose
from biovida.support_tools.support_tools import natural_key
from biovida.support_tools.support_tools import create_dir_if_needed
from biovida.support_tools.support_tools import list_to_bulletpoints
from biovida.support_tools.support_tools import InsufficientNumberOfFiles


# ----------------------------------------------------------------------------------------------------------
# Splitting Files into Train, Validation and Test
# ----------------------------------------------------------------------------------------------------------


def _subdirectories_in_path(path, to_block):
    """
    
    Generate a list of subdirectories in ``path``, excluding ``to_block``.

    :param path: a system path.
    :type path: ``str``
    :param to_block: a list of directories to exclude from the output.
    :type to_block: ``iterable``
    :return: see description.
    :rtype: ``list``
    """
    to_return = list()
    for i in os.listdir(path):
        if os.path.isdir(os.path.join(path, i)) and i not in to_block:
            to_return.append(os.path.join(path, i))
    return to_return


def _train_val_test_error_checking(data_dir, target_dir, action, delete_source, existing_files, tvt):
    """

    Check for possible errors for ``train_val_test()``.

    :param data_dir: see ``train_val_test()``.
    :type data_dir: ``str``
    :param target_dir: see ``train_val_test()``.
    :type target_dir: ``str``
    :param action: see ``train_val_test()``.
    :type action: ``str``
    :param delete_source: see ``train_val_test()``.
    :type delete_source: ``bool``
    :param existing_files: as evolved in side ``train_val_test()``.
    :type existing_files: ``dict``
    :param tvt: as evolved in side ``train_val_test()``.
    :type tvt: ``dict``
    """
    if not tvt:
        raise ValueError("None of `train`, `validation` and `test` are numeric.")
    for k, v in tvt.items():
        if v is True:
            raise ValueError("`{0}` cannot be `True`".format(k))
    if not isclose(sum(tvt.values()), 1):
        raise ValueError("The following parameters do not sum to 1: {0}.".format(", ".join(sorted(tvt.keys()))))
    if action not in ('copy', 'ndarray'):
        raise ValueError("`action` must be one of: 'copy', 'ndarray'.")
    if action == 'ndarray' and isinstance(target_dir, str):
        raise ValueError("`action` cannot equal 'ndarray' if `target_dir` is a string.")
    if not isinstance(delete_source, bool):
        raise TypeError("`delete_source` must be a boolean.")
    
    min_number_of_files = len(tvt.keys()) * len(existing_files.keys())
    for k, v in existing_files.items():
        if len(v) < min_number_of_files:
            raise InsufficientNumberOfFiles("\nThe '{0}' subdirectory in '{1}'\nonly contains {2} files, "
                                            "which is too few to distribute over {3} target locations.\n"
                                            "Calculation: "
                                            "len([{4}]) * len([{5}]) = {3}.".format(k, data_dir, len(v),
                                                                                    min_number_of_files,
                                                                                    ", ".join(tvt.keys()),
                                                                                    ", ".join(existing_files.keys())))


def _existing_files_dict_gen(directory, to_block):
    """

    Generate a dictionary of files in ``directory``.
    
    :param directory: a system path.
    :type directory: ``str``
    :param to_block: a list of subdirectory names to ignore.
    :type to_block: ``iterable``
    :return: a dictionary of the form: ``{subdirectory in ``directory``: [file_path, file_path, ...], ...}.
    :rtype: ``dict``
    """
    def list_dirs(path):
        return [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]

    def list_files(path):
        return [os.path.join(path, i) for i in os.listdir(path) if not i.startswith('.')]

    existing_files = dict()
    for d in list_dirs(directory):
        if d not in to_block:
            existing_files[d] = list_files(os.path.join(directory, d))
    return existing_files


def _list_divide(l, tvt):
    """

    Splits a list, ``l``, into proportions described in ``tvt``.

    :param l: a 'list' to be split according to ``tvt``.
    :type l: ``list``
    :param tvt: as evolved in side ``train_val_test()``.
    :type tvt: ``dict``
    :return: a dictionary of the form: ``{tvt_key_1: [file_path, file_path, ...], ...}``. Sorted to ensure
             generation order is train --> validation --> test.
    :rtype: ``tuple``

    :Example:

    >>> l = ['file/path/image_1.png', 'file/path/image_4.png', 'file/path/image_3.png', 'file/path/image_2.png']
    >>> tvt = {'validation': 0.5, 'train': 0.5}
    >>> _list_divide(l, tvt)
    ...
    {'train': ['file/path/image_1.png', 'file/path/image_2.png'],
    'validation': ['file/path/image_3.png', 'file/path/image_4.png']}

    """
    order_dict = {'train': 1, 'validation': 2, 'test': 3}

    # Natural sorting of items in ``l``.
    l_sorted = sorted(l, key=lambda x: natural_key(os.path.basename(x)))
    tvt_sorted = sorted(tvt.items(), key=lambda x: order_dict.get(x[0]))

    left, divided_dict = 0, dict()
    for e, (k, v) in enumerate(tvt_sorted, start=1):
        right = len(l_sorted) * v
        if e != len(tvt.keys()):
            right_full = int(left) + int(right)
        else:
            # Be greedy if on the last iteration.
            right_full = len(l_sorted)
        divided_dict[k] = l_sorted[int(left):right_full]
        left += right
    return sorted(divided_dict.items(), key=lambda x: order_dict.get(x[0]))


def _output_dict_with_ndarrays(dictionary):
    """

    Converts values (file paths) in the inner nest of ``dictionary`` to ``ndarray``s.

    :param dictionary: a nested dictionary, where values for the inner nest are file paths.
    :type dictionary: ``dict``
    :return: the values for the inner nest as replaced with ``ndarrays``.
    :rtype: ``dict``
    """
    return {k: {k2: np.array([imread(i) for i in v2]) for k2, v2 in v.items()} for k, v in dictionary.items()}


def train_val_test(data_dir,
                   train,
                   validation,
                   test,
                   target_dir=None,
                   action='copy',
                   delete_source=False,
                   verbose=True):
    """

    Splits data in ``data_dir`` into any combination of the following: ``train``, ``validation``, ``test``.

    :param data_dir: the directory containing the data. This directory should contain subdirectories (the categories)
                    populated with the files.

                .. warning::

                        Subdirectories entitled 'train', 'validation' and 'test' will be ignored.

    :type data_dir: ``str``
    :param train: the proportion images in ``data_dir`` to allocate to ``train``. If ``False`` or ``None``,
                  no images will be allocated.
    :type train: ``int``, ``float``, ``bool`` or ``None``
    :param validation: the proportion images in ``data_dir`` to allocate to ``validation``. If ``False``
                       or ``None``, no images will be allocated.
    :type validation: ``int``, ``float``, ``bool`` or ``None``
    :param test: the proportion images in ``data_dir`` to allocate to ``test``. If ``False`` or ``None``,
                 no images will be allocated.
    :type test: ``int``, ``float``, ``bool`` or ``None``
    :param target_dir: the location to output the images to (if ``action=True``). If ``None``, the output location will
                       be ``data_dir``. Defaults to ``None``.
    :type target_dir: ``str``
    :param action: one of: 'copy', 'ndarray'.

                    - if ``'copy'``: copy from files from ``data_dir`` to ``target_dir`` (default).
                    - if ``'ndarray'``: return a nested dictionary of ``ndarray`` ('numpy') arrays.

    :param delete_source: if ``True`` delete the source subdirectories in ``data_dir`` after copying is complete. Defaults to ``False``.

                          .. note::

                                This can be useful for transforming a directory 'inplace',
                                e.g., if ``data_dir`` and ``target_dir`` are the same and ``delete_source=True``.

    :type delete_source: ``bool``
    :param verbose: if ``True``, print the resultant structure. Defaults to ``True``.
    :type verbose: ``bool``
    :return:

        a dictionary of the form: ``{one of 'train', 'validation', 'test': {subdirectory in `data_dir`: [file_path, file_path, ...], ...}, ...}``.

        - if ``action='copy'``, the dictionary returned will be exactly as shown above.
        - if ``action='ndarray'``, 'file_path' will be replaced with the image as a ``ndarray`` and the list will be
                                    a ``ndarray``, e.g, ``array([matrix, matrix, ...])``.

    :rtype: ``dict``
    :raises ``ValueError``: if the combination of ``train``, ``validation``, ``test`` which which were passed
                            numeric values (i.e., ``int`` or ``float``) do not sum to 1.
    """
    groups = ('train', 'validation', 'test')
    target_path = data_dir if not isinstance(target_dir, str) else target_dir
    existing_dirs = _subdirectories_in_path(data_dir, to_block=groups)

    # Extract those of the train, validation, test (tvt) params which are numeric.
    tvt = {k: v for k, v in locals().items() if k in groups and isinstance(v, (float, int)) and not isinstance(v, bool)}

    # Generate a dictionary of files in `data_dir`
    existing_files = _existing_files_dict_gen(directory=data_dir, to_block=groups)

    # Check for invalid input
    _train_val_test_error_checking(data_dir=data_dir, target_dir=target_dir, action=action,
                                   delete_source=delete_source, existing_files=existing_files, tvt=tvt)

    output_dict = dict()
    for k, v in existing_files.items():
        for k2, v2 in _list_divide(v, tvt):
            if k2 not in output_dict:
                output_dict[k2] = {k: v2}
            else:
                output_dict[k2][k] = v2
            if action == 'copy':
                target = create_dir_if_needed(directory=os.path.join(target_path, os.path.join(k2, k)))
                for i in v2:
                    shutil.copy2(i, os.path.join(target, os.path.basename(i)))

    if verbose:
        print("\nStructure:\n")
        for k, v in output_dict.items():
            print("- {0}:\n{1}".format(k, list_to_bulletpoints(v)))

    if delete_source:
        for i in existing_dirs:
            shutil.rmtree(i)

    return output_dict if action == 'copy' else _output_dict_with_ndarrays(output_dict)








































