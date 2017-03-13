"""

    Utilities
    ~~~~~~~~~

"""
# Imports
import os
import shutil
import numpy as np
from warnings import warn
from scipy.ndimage import imread
from os.path import join as os_join

from biovida.support_tools.support_tools import isclose
from biovida.support_tools.support_tools import is_numeric
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
    return [os_join(path, i) for i in os.listdir(path) if os.path.isdir(os_join(path, i)) and i not in to_block]


def _train_val_test_error_checking(data, target_dir, action, delete_source, group_files_dict, tvt):
    """

    Check for possible errors for ``train_val_test()``.

    :param data: see ``train_val_test()``.
    :type data: ``str`` or ``dict``
    :param target_dir: see ``train_val_test()``.
    :type target_dir: ``str``
    :param action: see ``train_val_test()``.
    :type action: ``str``
    :param delete_source: see ``train_val_test()``.
    :type delete_source: ``bool``
    :param group_files_dict: as evolved in side ``train_val_test()``.
    :type group_files_dict: ``dict``
    :param tvt: as evolved in side ``train_val_test()``.
    :type tvt: ``dict``
    """
    if not tvt:
        raise ValueError("At least one of `train`, `validation` and `test` must be numeric.")
    for k, v in tvt.items():
        if v is True:
            raise ValueError("`{0}` cannot be `True`".format(k))
    if not isclose(sum(tvt.values()), 1):
        raise ValueError("The following parameters do not sum to 1: {0}.".format(", ".join(sorted(tvt.keys()))))
    if isinstance(data, dict) and not all(isinstance(i, (list, tuple)) for i in data.values()):
        raise TypeError("The values of `data` must be lists or tuples.")
    if action not in ('copy', 'ndarray'):
        raise ValueError("`action` must be one of: 'copy', 'ndarray'.")
    if action == 'ndarray' and isinstance(target_dir, str):
        warn("`target_dir` has no effect when `action='ndarray'`")
    if not isinstance(delete_source, bool):
        raise TypeError("`delete_source` must be a boolean.")
    
    min_number_of_files = len(tvt.keys()) * len(group_files_dict.keys())
    for k, v in group_files_dict.items():
        if len(v) < min_number_of_files:
            raise InsufficientNumberOfFiles("\nThe '{0}' subdirectory in '{1}'\nonly contains {2} files, "
                                            "which is too few to distribute over {3} target locations.\n"
                                            "Calculation: len([{4}]) * len([{5}]) = {3}.".format(
                                            k, data, len(v), min_number_of_files, ", ".join(tvt.keys()),
                                            ", ".join(group_files_dict.keys())))


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
    directories = [i for i in os.listdir(directory) if os.path.isdir(os_join(directory, i))]

    def list_files(path):
        return [os_join(path, i) for i in os.listdir(path) if not i.startswith('.')]

    return {d: list_files(os_join(directory, d)) for d in directories if d not in to_block}


def _list_divide(l, tvt):
    """

    Splits a list, ``l``, into proportions described in ``tvt``.

    :param l: a 'list' to be split according to ``tvt``.
    :type l: ``list``
    :param tvt: as evolved in side ``train_val_test()``.
    :type tvt: ``dict``
    :return: a dictionary of the form: ``{tvt_key_1: [file_path, file_path, ...], ...}``. Sorted to ensure
             generation order is train --> validation --> test.
    :rtype: ``list``

    .. note::

        This function will randomly shuffle ``l``.

    :Example:

    >>> l = ['file/path/image_1.png', 'file/path/image_4.png', 'file/path/image_3.png', 'file/path/image_2.png']
    >>> tvt = {'validation': 0.5, 'train': 0.5}
    >>> _list_divide(l, tvt)
    ...
    [('train', ['file/path/image_4.png', 'file/path/image_2.png']),
    ('validation', ['file/path/image_1.png', 'file/path/image_3.png'])]

    """
    order_dict = {'train': 1, 'validation': 2, 'test': 3}

    l_shuffled = np.random.permutation(l).tolist()
    tvt_sorted = sorted(tvt.items(), key=lambda x: order_dict.get(x[0]))

    left, divided_dict = 0, dict()
    for e, (k, v) in enumerate(tvt_sorted, start=1):
        right = len(l_shuffled) * v
        right_full = int(left) + int(right) if e != len(tvt.keys()) else len(l_shuffled)
        divided_dict[k] = l_shuffled[int(left):right_full]
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


def _train_val_test_engine(action, tvt, group_files_dict, target_path):
    """

    Engine to power ``train_val_test()``.

    :param action: see``train_val_test()``.
    :type action: ``str``
    :param tvt: as evolved in side ``train_val_test()``.
    :type tvt: ``dict``
    :param group_files_dict: as evolved in side ``train_val_test()``.
    :type group_files_dict: ``dict``
    :param target_path: see ``train_val_test()``.
    :type target_path: ``str``
    :return: a nested dictionary of the form ``{'train'/'val'/'test': {group_files_dict.key: [file, file, ...], ...}, ...}``.
    :rtype: ``dict``
    """
    output_dict = dict()
    for k, v in group_files_dict.items():
        for k2, v2 in _list_divide(v, tvt):
            if k2 not in output_dict:
                output_dict[k2] = {k: v2}
            else:
                output_dict[k2][k] = v2
            if action == 'copy':
                target = create_dir_if_needed(directory=os_join(target_path, os_join(k2, k)))
                for i in v2:
                    shutil.copy2(i, os_join(target, os.path.basename(i)))
    return output_dict


def train_val_test(data,
                   train,
                   validation,
                   test,
                   target_dir=None,
                   action='copy',
                   delete_source=False,
                   verbose=True):
    """

    Splits data in ``data`` into any combination of the following: ``train``, ``validation``, ``test``.

    :param data: 
    
        - a dictionary of the form: ``{group_name: [file_path, file_path], ...}``
        - the directory containing the data. This directory should contain subdirectories (the categories)
          populated with the files.

        .. warning::

                If a directory is passed, subdirectories therein entitled 'train', 'validation' and 'test'
                will be ignored.

    :type data: ``dict`` or ``str``
    :param train: the proportion images in ``data`` to allocate to ``train``. If ``False`` or ``None``,
                  no images will be allocated.
    :type train: ``int``, ``float``, ``bool`` or ``None``
    :param validation: the proportion images in ``data`` to allocate to ``validation``. If ``False``
                       or ``None``, no images will be allocated.
    :type validation: ``int``, ``float``, ``bool`` or ``None``
    :param test: the proportion images in ``data`` to allocate to ``test``. If ``False`` or ``None``,
                 no images will be allocated.
    :type test: ``int``, ``float``, ``bool`` or ``None``
    :param target_dir: the location to output the images to (if ``action=True``). If ``None``, the output location will
                       be ``data``. Defaults to ``None``.
    :type target_dir: ``str`` or ``None``
    :param action: one of: 'copy', 'ndarray'.

                    - if ``'copy'``: copy from files from ``data`` to ``target_dir`` (default).
                    - if ``'ndarray'``: return a nested dictionary of ``ndarray`` ('numpy') arrays.

    :type action: ``str``
    :param delete_source: if ``True`` delete the source subdirectories in ``data`` after copying is complete. Defaults to ``False``.

                          .. note::

                                This can be useful for transforming a directory 'in-place',
                                e.g., if ``data`` and ``target_dir`` are the same and ``delete_source=True``.

    :type delete_source: ``bool``
    :param verbose: if ``True``, print the resultant structure. Defaults to ``True``.
    :type verbose: ``bool``
    :return:

        a dictionary of the form: ``{one of 'train', 'validation', 'test': {subdirectory in `data`: [file_path, file_path, ...], ...}, ...}``.

        - if ``action='copy'``: the dictionary returned will be exactly as shown above.

        - if ``action='ndarray'``: 'file_path' will be replaced with the image as a ``ndarray`` and the list will be a ``ndarray``, e.g, ``array([matrix, matrix, ...])``.

    :rtype: ``dict``
    :raises ``ValueError``: if the combination of ``train``, ``validation``, ``test`` which which were passed
                            numeric values (i.e., ``int`` or ``float``) do not sum to 1.


    .. note::

          Files are randomly shuffled prior to assignment.

    .. warning::

           In the case of division with a remainder, preference is as follows: `train` < `validation` < `test`.
           For instance, if we have ``train=0.7``, ``validation=0.3``, and the number of files in a given subdirectory
           equal to ``6`` (as in the final example below) the number of files allocated to `train` will be rounded
           down, in favor of `validation` obtaining the final file. In this instance, `train` would obtain
           `4` files (``floor(0.7 * 6) = 4``) and `validation` would obtain 2 files (``ceil(0.3 * 6) = 2``).


    :Example:

    The examples below use a sample directory entitled ``images``.

    This is its structure:

    .. code-block:: bash

        $ tree /path/to/data/images
        ├── ct
        │   ├── ct_1.png
        │   ├── ct_2.png
        │   ├── ct_3.png
        │   ├── ct_4.png
        │   ├── ct_5.png
        │   └── ct_6.png
        └── mri
            ├── mri_1.png
            ├── mri_2.png
            ├── mri_3.png
            ├── mri_4.png
            ├── mri_5.png
            └── mri_6.png

    |
    | **Usage 1**: Obtaining ndarrays

    >>> from biovida.support_tools import train_val_test
    >>> tt = train_val_test(data='/path/to/data/images', train=0.7, validation=None, test=0.3,
    ...                     action='ndarray')

    The resultant ndarrays can be unpacked into objects as follows:

    >>> train_ct, train_mri = tt['train']['ct'], tt['train']['mri']
    >>> test_ct, test_mri = tt['test']['ct'], tt['test']['mri']

    |
    | **Usage 2**: Reorganize a Directory In-place

    >>> from biovida.support_tools import train_val_test
    >>> tv = train_val_test(data='/path/to/data/images', train=0.7, validation=0.3, test=None,
    ...                     action='copy', delete_source=True)

    Which results in the following structure:

    .. code-block:: bash

        $ tree /path/to/data/images
        ├── train
        │   ├── ct
        │   │   ├── ct_4.png
        │   │   ├── ct_5.png
        │   │   ├── ct_3.png
        │   │   └── ct_1.png
        │   └── mri
        │       ├── mri_2.png
        │       ├── mri_1.png
        │       ├── mri_5.png
        │       └── mri_4.png
        └── validation
            ├── ct
            │   ├── ct_2.png
            │   └── ct_6.png
            └── mri
                ├── mri_3.png
                └── mri_6.png

    .. note::

        The following changes to **Usage 2** would preserve the original ``ct`` and ``mri`` directories:

        - setting ``delete_source=False`` (the default) and/or
        - providing a path to ``target_dir``, e.g., ``target_dir='/path/to/output/output_data'``

    """
    groups = ('train', 'validation', 'test')
    if action == 'copy':
        if isinstance(target_dir, str):
            target_path = target_dir
        elif isinstance(data, str) and not isinstance(target_dir, str):
            target_path = data
            warn("`target_path` is not a string; using `data` as the output location")
        else:
            raise TypeError("`target_dir` must be a system path if `data` is not.")
    else:
        target_path = None

    existing_dirs = _subdirectories_in_path(data, to_block=groups) if isinstance(data, str) else None

    # Extract those of the train, validation, test (tvt) params which are numeric.
    tvt = {k: v for k, v in locals().items() if k in groups and is_numeric(v)}

    if isinstance(data, str):
        group_files_dict = _existing_files_dict_gen(directory=data, to_block=groups)
    elif isinstance(data, dict):
        group_files_dict = data
    else:
        raise TypeError("`data` must be a string or dictionary.")

    _train_val_test_error_checking(data=data, target_dir=target_dir, action=action,
                                   delete_source=delete_source, group_files_dict=group_files_dict, tvt=tvt)

    output_dict = _train_val_test_engine(action=action, tvt=tvt,
                                         group_files_dict=group_files_dict, target_path=target_path)

    if verbose:
        print("\nStructure:\n")
        for k, v in output_dict.items():
            print("- {0}:\n{1}".format(k, list_to_bulletpoints(v)))

    if existing_dirs is not None and delete_source:
        for i in existing_dirs:
            shutil.rmtree(i)

    return output_dict if action == 'copy' else _output_dict_with_ndarrays(output_dict)

























