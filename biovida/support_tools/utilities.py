"""

    Utilities
    ~~~~~~~~~

"""
# Imports
import os
import shutil
from math import ceil
from scipy.ndimage import imread

from biovida.support_tools.support_tools import create_dir_if_needed
from biovida.support_tools.support_tools import list_to_bulletpoints


# ----------------------------------------------------------------------------------------------------------
# Splitting into Train, Validation & Test
# ----------------------------------------------------------------------------------------------------------


class InsufficientNumberOfFiles(Exception):
    pass


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


def _train_val_test_error_checking(data_dir,
                                   train,
                                   validation,
                                   test,
                                   target_dir,
                                   action,
                                   delete_source,
                                   existing_files,
                                   tvt):
    """

    :param data_dir:
    :param train:
    :param validation:
    :param test:
    :param target_dir:
    :param action:
    :param delete_source:
    :param existing_files:
    :param tvt:
    :return:
    """
    if not tvt:
        raise ValueError("None of `train`, `validation` and `test` are numeric.")
    for k, v in tvt.items():
        if v is True:
            raise ValueError("`{0}` cannot be `True`".format(k))
    if sum(tvt.values()) != 1:
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
                                            "Calculation: ({4}) * ({5}) = {3}.".format(k, data_dir, len(v),
                                                                                       min_number_of_files,
                                                                                       ", ".join(tvt.keys()),
                                                                                       ", ".join(existing_files.keys())))


def _existing_files_dict_gen(directory, groups):
    """

    Generate a dictionary of files in ``directory``.
    
    :param directory:
    :param groups: 
    :return: 
    """
    def list_dirs(path):
        return [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]

    def list_files(path):
        return [os.path.join(path, i) for i in os.listdir(path) if not i.startswith('.')]

    existing_files = dict()
    for d in list_dirs(directory):
        if d not in groups:
            existing_files[d] = list_files(os.path.join(directory, d))
    return existing_files


def _list_divide(l, tvt):
    """

    :param l:
    :param tvt:
    :return:
    :rtype: ``dict``
    """
    left, divided_dict = 0, dict()
    for e, (k, v) in enumerate(tvt.items()):
        right = len(l) * v
        # Be greedy if on the last key
        right_rounded = ceil(right) if e == len(tvt.keys()) else int(right)
        divided_dict[k] = l[int(left):right_rounded]
        left += right
    return divided_dict


def _output_dict_with_ndarrays(output_dict):
    """

    :param output_dict:
    :type output_dict: ``dict``
    :return:
    :rtype: ``dict``
    """
    return {k: {k2: [imread(i) for i in v2] for k2, v2 in v.items()} for k, v in output_dict.items()}


def train_val_test(data_dir,
                   train,
                   validation,
                   test,
                   target_dir=None,
                   action='copy',
                   delete_source=False,
                   verbose=True):
    """

    :param data_dir: the directory containing the data. This directory should contain subdirectories (the categories)
                    populated with the files.
    :type data_dir: ``str``
    :param train: the proportion images in ``data_dir`` to allocate to the ``train`` folder. If ``False`` or ``None``,
                  no images will be allocated to the ``train`` folder.
    :type train: ``int``, ``float``, ``bool`` or ``None``
    :param validation: the proportion images in ``data_dir`` to allocate to the ``validation`` folder. If ``False``
                       or ``None``, no images will be allocated to the ``validation`` folder.
    :type validation: ``int``, ``float``, ``bool`` or ``None``
    :param test: the proportion images in ``data_dir`` to allocate to the ``test`` folder. If ``False`` or ``None``,
                 no images will be allocated to the ``test`` folder.
    :type test: ``int``, ``float``, ``bool`` or ``None``
    :param target_dir: the location to output the images to (if ``action=True``).
    :type target_dir: ``str``
    :param action: one of:
                    - 'copy': to copy from files from ``data_dir`` to ``target_dir``
                    - 'ndarray': to return a nested dictionary of ``ndarray`` ('numpy') arrays.
    :param delete_source: if ``True`` delete the source subdirectories in ``data_dir``.
    :type delete_source: ``bool``
    :param verbose: if ``True``, print additional details.
    :type verbose: ``bool``
    :return: if ``action='copy'``, a dictionary of the form: ``{one of train, validation, test: {subdirectory in `data_dir`: [file_path, file_path, ...], ...}, ...}``.
             if ``action='ndarray'``, 'file_path' will be replaced with the images as a ``ndarray``.
    :rtype: ``dict``
    """
    groups = ('train', 'validation', 'test')
    target_path = data_dir if not isinstance(target_dir, str) else target_dir
    existing_dirs = _categories_in_dir(data_dir, to_block=groups)

    # Extract those of train, validation, test which are numeric
    c = {k: v for k, v in locals().items() if k in groups and isinstance(v, (float, int)) and not isinstance(v, bool)}

    # Generate a dictionary of files in `data_dir`
    existing_files = _existing_files_dict_gen(data_dir=data_dir, groups=groups)

    # Check for invlaid input
    _train_val_test_error_checking(data_dir=data_dir, train=train, validation=validation, test=test,
                                   target_dir=target_dir, action=action, delete_source=delete_source,
                                   existing_files=existing_files, tvt=tvt)

    output_dict = dict()
    for k, v in existing_files.items():
        for k2, v2 in _list_divide(v, tvt).items():
            if k2 not in output_dict:
                output_dict[k2] = {k: v2}
            else:
                output_dict[k2][k] = v2
            if action == 'copy':
                target = create_dir_if_needed(directory=os.path.join(target_path, os.path.join(k2, k)))
                for i in v2:
                    shutil.copy2(i, os.path.join(target, os.path.basename(i)))

    if verbose:
        print("Structure:\n")
        for k, v in output_dict.items():
            print("- {0}:\n{1}".format(k, list_to_bulletpoints(v)))

    if delete_source:
        for i in existing_dirs:
            shutil.rmtree(i)

    return output_dict if action == 'copy' else _output_dict_with_ndarrays(output_dict)








































