"""

    Image Cache Management
    ~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import os
import shutil
import numpy as np
import pandas as pd
from warnings import warn
from itertools import chain
from collections import Counter

# General Image Support Tools
from biovida.images._image_tools import ActionVoid

# General Support Tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import multimap
from biovida.support_tools.support_tools import directory_existence_handler

# Utilities
from biovida.support_tools.utilities import train_val_test

# Import Printing Tools
from biovida.support_tools import pandas_pprint


# ----------------------------------------------------------------------------------------------------------
# Relationship Mapping Function
# ----------------------------------------------------------------------------------------------------------


def _openi_image_relation_map(data_frame):
    """

    Algorithm to find the index of rows which reference
    the same image in the cache (the image size can vary).

    :param data_frame:
    :type data_frame: ``Pandas DataFrame``
    :return:
    :rtype: ``Pandas DataFrame``
    """
    # Copy the data_frame
    df = data_frame.copy(deep=True)

    # Reset the index
    df = df.reset_index(drop=True)

    # Get duplicated img_large occurrences. Use of 'img_large' is arbitrary, could have used
    # any of the 'img_...' columns, e.g., 'img_thumb' or 'img_grid150'.
    duplicated_image_refs = (k for k, v in Counter(df['img_large']).items() if v > 1)

    # Get the indices of duplicates
    dup_index = {k: df[df['img_large'] == k].index.tolist() for k in duplicated_image_refs}

    def related(image_large, index):
        """Function to look for references to the same image in the cache."""
        if image_large in dup_index:
            return tuple(sorted([i for i in dup_index[image_large] if i != index]))
        else:
            return np.NaN

    # Apply `relate()`
    df['shared_image_ref'] = [related(image, index) for image, index in zip(df['img_large'], df.index)]

    return df


# ----------------------------------------------------------------------------------------------------------
# Merging New Records with the Existing Cache
# ----------------------------------------------------------------------------------------------------------


def _dict_to_tot(d):
    """

    Convert a dictionary to a tuple of tuples and sort by the former keys.

    :param d: any dictionary.
    :type d: ``dict``
    :return: ``d`` as a tuple of tuples, sorted by the former key values.
    :rtype: ``tuple``
    """
    if not isinstance(d, dict):
        return d

    def values_to_tuples(d):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
    return tuple(sorted(values_to_tuples(d).items(), key=lambda x: x[0]))


def _record_update_dbs_joiner(records_db, update_db):
    """

    Join and drop rows for which `update_db`'s columns exclusively contain NaNs.

    :param records_db: permanent database/dataframe which keeps a record of files in the cache.
    :type records_db: ``Pandas DataFrame``
    :param update_db: database/dataframe to 'update' ``records_db``
    :type update_db: ``Pandas DataFrame``
    :return: ``records_db`` with ``update_db`` left-joined.
    :rtype: ``Pandas DataFrame``
    """
    joined_db = records_db.join(update_db, how='left').fillna(np.NaN).dropna(subset=list(update_db.columns), how='all')
    return joined_db.reset_index(drop=True)


def _load_temp_dbs(temp_db_path):
    """

    Load temporary databases in the 'databases/__temp__' directory.

    :param temp_db_path: path to the temporary databases (must be pickled and use the '.p' extension).
    :type temp_db_path: ``str``
    :return: all of the '.p' dataframes in ``temp_db_path`` merged into a single dataframe.
    :rtype: ``Pandas DataFrame`` or ``None``
    """
    # Get pickled objects in ``temp_db_path``
    db_paths = [os.path.join(temp_db_path, p) for p in os.listdir(temp_db_path) if p.endswith(".p")]

    if not len(db_paths):
        return None

    # Set of to group db_paths by
    unique_pull_times = {os.path.basename(path).split("__")[0] for path in db_paths}

    groupings = list()
    for pull_time in unique_pull_times:
        group = [p for p in db_paths if pull_time in p]
        if len(group) == 1:
            os.remove(group[0])
            warn("FileNotFoundError: Either '{0}__records_db.p' or '{0}__update_db.p'"
                 " is missing from\n {1}\nDeleting the file which is present...\n"
                 "As a result, images obtained from the last `pull()` will likely be missing"
                 " from `cache_records_db`.\nFor this reason, it is recommended that you\n"
                 "**precisely** repeat your last `search()` and `pull()`.".format(pull_time, temp_db_path))
        else:
            groupings.append(group)

    if not len(groupings):
        return None

    # Read the dataframes in the '__temp__' directory into memory
    frames = list()
    for group in groupings:
        records_db = pd.read_pickle([i for i in group if "__records_db.p" in i][0])
        update_db = pd.read_pickle([i for i in group if "__update_db.p" in i][0])
        frames.append(_record_update_dbs_joiner(records_db, update_db))

    # Concatenate all frames
    return pd.concat(frames, ignore_index=True)


def _relationship_mapper(data_frame, interface_name):
    """

    :param data_frame:
    :type data_frame: ``Pandas DataFrame``
    :param interface_name:
    :type interface_name: ``str``
    :return:
    :rtype: ``Pandas DataFrame``
    """
    _relationship_mapping_dict = {
        # Keys: Interface Class Name.
        # Values: mapping function
        'OpeniInterface': _openi_image_relation_map
    }

    if interface_name in _relationship_mapping_dict:
        relationship_mapping_func = _relationship_mapping_dict[interface_name]
        data_frame = relationship_mapping_func(data_frame)

    return data_frame


def _records_db_merge(interface_name,
                      current_records_db,
                      records_db_update,
                      columns_with_dicts,
                      duplicates_subset_columns,
                      rows_to_conserve_func=None,
                      post_concat_mapping=None,
                      columns_with_iterables_to_sort=None):
    """

    Merge the existing record database with new additions.

    .. warning::

        Both ``current_records_db`` and ``records_db_update`` are expected to have 'pull_time' columns.

    :param current_records_db: the existing record database.
    :type current_records_db: ``Pandas DataFrame``
    :param records_db_update: the new records dataframe to be merged with the existing one (``current_records_db``).
    :type records_db_update: ``Pandas DataFrame``
    :param columns_with_dicts: a list of columns which contain dictionaries. Note: this column *should* contain only
                               dictionaries or NaNs.
    :type columns_with_dicts: ``list``, ``tuple`` or ``None``.
    :param duplicates_subset_columns: a list (or tuple) of columns to consider when dropping duplicates.

                    .. warning::

                            Do *not* include the 'shared_image_ref' column as this function will recompute it.

    :type duplicates_subset_columns: ``list`` or ``tuple``
    :param rows_to_conserve_func: function to generate a list of booleans which denote whether or not the image is,
                                  in fact, present in the cahce. If not, remove it from the database to be saved.
    :type rows_to_conserve_func: ``function``
    :param columns_with_iterables_to_sort: columns which themselves contain lists or tuples which should be sorted
                                  prior to dropping. Defaults to ``None``.
    :type columns_with_iterables_to_sort: ``list`` or ``tuple``
    :return: a dataframe which merges ``current_records_db`` and ``records_db_update``
    :rtype: ``Pandas DataFrame``
    """
    # Note: this function does not explicitly handle cases where combined_dbs has length 0, no obvious need to though.

    # Load in the current database and combine with the `records_db_update` database
    combined_dbs = pd.concat([current_records_db, records_db_update], ignore_index=True)

    # Mark each row to conserve order following ``pandas.drop_duplicates()``.
    combined_dbs['__temp_order__'] = range(combined_dbs.shape[0])

    # Analyze which rows to drop
    if rows_to_conserve_func is not None:
        combined_dbs = combined_dbs[combined_dbs.apply(rows_to_conserve_func, axis=1)]

    # Apply post merge mapping
    if isinstance(post_concat_mapping, (list, tuple)) and len(post_concat_mapping) == 3:
        column_name, column_to_extract, func = post_concat_mapping
        combined_dbs[column_name] = func(combined_dbs[column_to_extract].tolist())

    # Note: Typically these will be 'in sync'. However, if they are not, preference is given
    # to 'biovida_version' s.t. the data harvested with the latest version is given preference.
    combined_dbs = combined_dbs.sort_values(['biovida_version', 'pull_time'])

    # Convert items in ``columns_with_dicts`` from dictionaries to tuple of tuples.
    # (making them hashable, as required by ``pandas.drop_duplicates()``).
    combined_dbs = multimap(combined_dbs, columns=columns_with_dicts, func=_dict_to_tot)

    # Sort iterables in columns with iterables
    combined_dbs = multimap(combined_dbs, columns=columns_with_iterables_to_sort, func=lambda x: tuple(sorted(x)))

    # Drop Duplicates (keeping the most recent).
    combined_dbs = combined_dbs.drop_duplicates(subset=duplicates_subset_columns, keep='last')

    # Convert the tuples back to dictionaries
    combined_dbs = multimap(combined_dbs, columns=columns_with_dicts, func=dict)

    # Sort against the original order.
    combined_dbs = combined_dbs.sort_values('__temp_order__')

    # Map relationships in the dataframe.
    combined_dbs = _relationship_mapper(data_frame=combined_dbs, interface_name=interface_name)

    return combined_dbs.drop('__temp_order__', axis=1).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------
# Pruning the Cache of Deleted Files
# ----------------------------------------------------------------------------------------------------------


def _files_existence_checker(to_check):
    """

    Checks if ``to_check`` exists.
    If not take the following action:

    - If a ``str``, return ``to_check`` if it exists else ``None``.

    - If a ``list`` or ``tuple``, remove items that do not exist.
      If resultant length is zero return ``None``.

    :param to_check: file, or iterable of file, to check the existence of
    :type to_check: ``str``, ``list`` or ``tuple``
    :return: ``to_check``, pruned ``to_check`` (if iterable) or ``None`` (all files removed).
    :rtype: ``str``, ``list``, ``tuple``, ``None`` or ``type(to_check)``
    """
    if isinstance(to_check, str):
        return to_check if os.path.isfile(to_check) else None
    elif isinstance(to_check, (list, tuple)):
        files_present = tuple([i for i in to_check if os.path.isfile(i)])
        return files_present if len(files_present) else None
    else:
        return to_check


def _df_pruner(cache_records_db, columns):
    """

    Prune ``cache_records_db`` by reviewing ``columns``.

    :param cache_records_db: see ``_prune_rows_with_deleted_images()``.
    :type cache_records_db: ``Pandas DataFrame``
    :param columns: see ``_prune_rows_with_deleted_images()``.
    :type columns: ``list``
    :return: a pruned ``cache_records_db``.
    :rtype: ``Pandas DataFrame``
    """
    for c in columns:
        cache_records_db[c] = cache_records_db[c].map(_files_existence_checker)

    # Mark rows to remove
    indices_to_drop = cache_records_db[columns].apply(lambda x: not all(x[i] is None for i in columns), axis=1)

    # Drop and reset the index
    return cache_records_db[indices_to_drop].fillna(np.NaN).reset_index(drop=True)


def _prune_rows_with_deleted_images(cache_records_db, columns, save_path):
    """

    Tool to remove reference to images that have been manually deleted from the cache.
    After this pruning has occurred, ``cache_records_db`` is saved at ``save_path``.

    If a column element is a string, it will be left 'as is' if the file exists,
    otherwise it the entire row will be marked for deletion.

    If a column element is a tuple, image paths in the tuple that do not exist will be removed from the
    tuple. If the resultant tuple is of zero length (i.e., all images have been deleted), the entire row
    will be marked for deletion.

    If, for a given row, all entries for the columns in ``columns`` are ``None`` (i.e., the images
    have been deleted), that row will be removed from ``cache_records_db``.
    Note: if one column is marked for deletion and another is not, the row will be conserved.

    .. note::

        If no images have been deleted, the output of this function will be the same as the input.

    :param cache_records_db: a cache_records_db from the ``OpeniInterface()`` or ``CancerImageInterface()``
    :type cache_records_db: ``Pandas DataFrame``
    :param columns: a ``list`` of columns with paths to cached images. These columns can be columns of
                    strings or columns of tuples.

        .. warning::

            This parameter *must* be a ``list``.

    :type columns: ``list``
    :param save_path: the location to save ``cache_records_db``.
    :type save_path: ``str``
    :return: a pruned ``cache_records_db``
    :rtype: ``Pandas DataFrame``
    """
    pruned_cache_records_db = _df_pruner(cache_records_db, columns)
    pruned_cache_records_db.to_pickle(save_path)
    return pruned_cache_records_db


# ----------------------------------------------------------------------------------------------------------
# Interface Data
# ----------------------------------------------------------------------------------------------------------


_image_instance_image_columns = {
    # Note: the first item should be the default.
    'OpeniInterface': ('cached_images_path',),
    'CancerImageInterface': ('cached_images_path', 'cached_dicom_images_path'),
    'unify_against_images': ('cached_images_path',),
}


# ----------------------------------------------------------------------------------------------------------
# Deleting Image Data
# ----------------------------------------------------------------------------------------------------------


def _robust_delete(to_delete):
    """

    Function to delete ``to_delete``.
    If a list (or tuple), all paths therein will be deleted.

    :param to_delete: a file, or multiple files to delete. Note: if ``to_delete`` is not a ``string``,
                     ``list`` or ``tuple``, no action will be taken.
    :type to_delete: ``str``, ``list``  or ``tuple``
    """
    def delete_file(td):
        if os.path.isfile(td):
            os.remove(td)

    if isinstance(to_delete, str):
        delete_file(to_delete)
    elif isinstance(to_delete, (list, tuple)):
        for t in to_delete:
            if isinstance(t, str):
                delete_file(t)


def _double_check_with_user():
    """

    Ask the user to verify they wish to proceed.

    """
    response = input("This action cannot be undone.\n"
                     "Do you wish to continue (y/n)?")
    if cln(response).lower() not in ('yes', 'ye', 'es', 'y'):
        raise ActionVoid("\n\nAction Canceled.")


def _pretty_print_image_delete(deleted_rows, verbose):
    """

    Pretty print ``deleted_rows``.

    :param deleted_rows: as evolved inside ``image_delete``.
    :type deleted_rows: ``dict``
    :param verbose: see ``image_delete``.
    :type verbose: ``bool``
    """
    if deleted_rows and verbose:
        print("\nSummary of Deleted Rows:\n")
        to_print = pd.DataFrame.from_dict(deleted_rows, orient='index').T  # handles when values are of unequal length.
        pandas_pprint(to_print[sorted(to_print.columns, reverse=True)], full_rows=True, suppress_index=True)


def image_delete(instance, delete_rule, verbose=True):
    """

    Delete images from the cache.

    .. warning::

        The effects of this function can only be undone by downloading the deleted data again.

    :param instance: an instance of ``OpeniInterface`` or ``CancerImageInterface``.
    :type instance: ``OpeniInterface`` or ``CancerImageInterface``
    :param delete_rule: must be one of: ``'all'`` (delete *all* data) or a ``function`` which (1) accepts a single
                        parameter (argument) and (2) returns ``True`` when the data is to be deleted.
    :type delete_rule: ``str`` or ``function``
    :param verbose: if ``True``, print a
    :type verbose: ``bool``
    :return: a dictionary of the indices which were dropped. Example: ``{'records_db': [58, 59], 'cache_records_db': [158, 159]}``.
    :rtype: ``dict``

    :Example:

    >>> from biovida.images import image_delete
    >>> from biovida.images import OpeniInterface
    ...
    >>> opi = OpeniInterface()
    >>> opi.search(image_type=['ct', 'mri'], collection='medpix')
    >>> opi.pull()
    ...
    >>> def my_delete_rule(row):
    >>>     if isinstance(row['abstract'], str) and 'Oompa Loompas' in row['abstract']:
    >>>         return True
    ...
    >>> image_delete(opi, delete_rule=my_delete_rule)

    .. note::

        In this example, any rows in the ``records_db`` and ``cache_records_db``
        for which the 'abstract' column contains the string 'Oompa Loompas' will be deleted.
        Any images associated with this row will also be destroyed.

    .. warning::

        If a function is passed to ``delete_rule`` it *must* return a boolean ``True`` to delete a row.
        **All other output will be ignored**.

    """
    index_dict = dict()
    _double_check_with_user()

    delete_all = False
    if isinstance(delete_rule, str):
        if cln(delete_rule).lower() == 'all':
            delete_all = True
        else:
            raise ValueError("`delete_rule` must be 'all' or a `function`.")

    def delete_rule_wrapper(row, enact):
        """Wrap delete_rule to ensure the output is a boolean."""
        do_delete = delete_rule(row) if callable(delete_rule) else False
        if delete_all or (isinstance(do_delete, bool) and do_delete is True):
            if enact:
                for c in _image_instance_image_columns[instance.__class__.__name__]:
                    _robust_delete(row[c])
            return False  # drop from the dataframe
        else:
            return True   # keep in the dataframe

    if isinstance(instance.records_db, pd.DataFrame):
        index_dict['records_db'] = {'before': instance.records_db.index.tolist()}
        to_conserve = instance.records_db.apply(lambda r: delete_rule_wrapper(r, enact=False), axis=1)
        instance.records_db = instance.records_db[to_conserve.tolist()].reset_index(drop=True)
        index_dict['records_db']['after'] = instance.records_db.index.tolist()
    if isinstance(instance.cache_records_db, pd.DataFrame):
        index_dict['cache_records_db'] = {'before': instance.cache_records_db.index.tolist()}
        # Apply ``delete_rule`` to ``cache_records_db``.
        _ = instance.cache_records_db.apply(lambda r: delete_rule_wrapper(r, enact=True), axis=1)
        # Prune ``cache_records_db`` by inspecting which images have been deleted.
        instance._load_prune_cache_records_db(load=False)
        # Map relationships, if applicable.
        instance.cache_records_db = _relationship_mapper(instance.cache_records_db, instance.__class__.__name__)
        # Save the updated ``cache_records_db`` to 'disk'.
        instance._save_cache_records_db()
        index_dict['cache_records_db']['after'] = instance.cache_records_db.index.tolist()
    else:
        raise TypeError("`cache_record_db` is not a DataFrame.")

    deleted_rows = {k: sorted(set(v['before']) - set(v['after'])) for k, v in index_dict.items()}
    _pretty_print_image_delete(deleted_rows=deleted_rows, verbose=verbose)

    return deleted_rows


# ----------------------------------------------------------------------------------------------------------
# Divvy Image Data
# ----------------------------------------------------------------------------------------------------------


def _image_divvy_error_checking(action, train_val_test_dict):
    """

    Check for possible errors for ``image_divvy()``.

    :param train_val_test_dict: see ``image_divvy()``.
    :type action: ``str``
    :param train_val_test_dict: see ``image_divvy()``.
    :type train_val_test_dict: ``dict``
    """
    if isinstance(train_val_test_dict, dict):
        if not train_val_test_dict:
            raise KeyError("`train_val_test_dict` is empty")
        if action == 'copy' and 'target_dir' not in train_val_test_dict:
            raise KeyError("`train_val_test_dict` must contain a `target_dir` key "
                           "if `action='copy'`.")
        for k in train_val_test_dict:
            if k not in ('train', 'validation', 'test', 'target_dir'):
                raise KeyError("Invalid `train_val_test_dict` key: '{0}'.")
        if action == 'ndarray' and'target_dir' in train_val_test_dict:
            warn("The 'target_dir' key in `train_val_test_dict` has no "
                 "effect when `action='ndarray'`.")


def _robust_copy(to_copy, copy_to, allow_creation, allow_overwrite):
    """

    Function to copy ``to_copy``.
    If a list (or tuple), all paths therein will be copied.

    :param to_copy: a file, or multiple files to delete. Note: if ``to_copy`` is not a ``string``,
                     ``list`` or ``tuple``, no action will be taken.
    :type to_copy: ``str``, ``list``  or ``tuple``
    :param copy_to: the location for the image
    :type copy_to: ``str``
    :param allow_creation: if ``True``, create ``path_`` if it does not exist, else raise.
    :type allow_creation: ``bool``
    :param allow_overwrite: if ``True`` allow existing images to be overwritten. Defaults to ``True``.
    :type allow_overwrite: ``bool``
    """
    directory_existence_handler(path_=copy_to, allow_creation=allow_creation)

    def copy_util(from_path):
        if os.path.isfile(from_path):
            to_path = os.path.join(copy_to, os.path.basename(from_path))
            if not allow_overwrite and os.path.isfile(to_path):
                raise FileExistsError("The following file already exists:\n{0}".format(to_path))
            shutil.copy2(from_path, to_path)
        else:
            warn("No such file:\n'{0}'".format(from_path))

    if isinstance(to_copy, str):
        copy_util(from_path=to_copy)
    elif isinstance(to_copy, (list, tuple)):
        for c in to_copy:
            if isinstance(c, str):
                copy_util(from_path=c)


def _divvy_column_selector(instance, db_to_extract, image_column, data_frame):
    """

    Select the column to use when copying images from.

    :param instance:  see ``image_divvy()``
    :type instance: ``OpeniInterface`` or ``CancerImageInterface``
    :param db_to_extract: see ``image_divvy()``
    :type db_to_extract: ``str``
    :param image_column: see ``image_divvy()``
    :type image_column: ``str``
    :param data_frame: as evolved inside  ``image_divvy()``.
    :type data_frame: ``Pandas DataFrame``
    :return: the column in ``data_frame`` to use when copying images to the new location.
    :rtype: ``str``
    """
    if image_column is None:
        return _image_instance_image_columns[instance.__class__.__name__][0]
    elif not isinstance(image_column, str):
        raise TypeError('`image_column` must be a string or `None`.')
    elif image_column in _image_instance_image_columns[instance.__class__.__name__]:
        if image_column not in data_frame.columns:
            raise KeyError("The '{0}' column is missing from '{1}'.".format(image_column, db_to_extract))
        return image_column
    else:
        raise KeyError("'{0}' is not a valid image column for '{1}'.".format(image_column, db_to_extract))


def _image_divvy_wrappers_gen(divvy_rule, action, train_val_test_dict, column_to_use, create_dirs, allow_overwrite):
    """

    Wrap the ``divvy_rule`` passed to ``image_divvy()``.

    :param divvy_rule: see ``image_divvy()``.
    :type divvy_rule: ``str`` or ``function``
    :param action: see ``image_divvy()``.
    :type action: ``str``
    :param train_val_test_dict:  see ``image_divvy()``.
    :type train_val_test_dict: ``dict``
    :param column_to_use:  as evolved inside ``image_divvy`` by ``_divvy_column_selector()``
    :type column_to_use: ``str``
    :param create_dirs: see ``image_divvy()``.
    :type create_dirs: ``bool``
    :param allow_overwrite: see ``image_divvy()``.
    :type allow_overwrite: ``bool``
    :return: a function which wraps the function passed to ``divvy_rule()``.
    :rtype: ``function``
    """
    # ToDo: handle action='ndarray' and train_val_test_dict=None.

    def copy_rule_wrapper(row, copy_to):
        if isinstance(copy_to, (str, tuple, list)):
            all_copy_targets = [copy_to] if isinstance(copy_to, str) else copy_to
            if not len(all_copy_targets):
                return None
            for i in all_copy_targets:
                if not isinstance(i, str):
                    raise TypeError("`divvy_rule` returned an iterable containing a element which is not a string.")
                _robust_copy(to_copy=row[column_to_use], copy_to=i,
                             allow_creation=create_dirs, allow_overwrite=allow_overwrite)
        elif copy_to is not None:
            raise TypeError("String, list or tuple expected. "
                            "`divvy_rule` returned an object of type '{0}'.".format(type(copy_to).__name__))

    def divvy_rule_wrapper(row):
        if not isinstance(divvy_rule, str) and not callable(divvy_rule):
            raise TypeError("`divvy_rule` must be a string or function.")

        copy_to = divvy_rule(row) if callable(divvy_rule) else divvy_rule
        if not isinstance(copy_to, (str, list, tuple)):
            return None
        if not len(copy_to):
            return None
        if action == 'copy' and not isinstance(train_val_test_dict, dict):
            copy_rule_wrapper(row, copy_to)
            return None
        elif isinstance(row[column_to_use], (str, tuple, list)):
            cache_info = [row[column_to_use]] if isinstance(row[column_to_use], str) else list(row[column_to_use])
            if isinstance(copy_to, str):
                return [[os.path.basename(copy_to), cache_info]]
            elif isinstance(copy_to, (list, tuple)):
                return [[os.path.basename(c), cache_info] for c in copy_to]
        else:
            return None

    return divvy_rule_wrapper


def _image_divvy_train_val_test_wrapper(action, verbose, divvy_info, train_val_test_dict):
    """

    Take ``divvy_info`` and pass to ``train_val_test()``

    :param action: see ``image_divvy()``.
    :type action: ``str``
    :param verbose: see ``image_divvy()``.
    :type verbose: ``bool``
    :param divvy_info: as evolved inside ``image_divvy``.
    :type divvy_info: ``Pandas DataFrame``
    :param train_val_test_dict see ``image_divvy()``.
    :type train_val_test_dict:
    :return: see ``train_val_test``.
    :rtype: ``dict``
    """
    divvy_info_groupby = divvy_info.dropna().groupby(0).apply(lambda x: x[1].tolist())
    divvy_info_dict = {k: list(chain(*v)) for k, v in divvy_info_groupby.to_dict().items()}

    target = train_val_test_dict.get('target_dir') if action == 'copy' else None
    output_dict = train_val_test(data=divvy_info_dict,
                                 train=train_val_test_dict.get('train', None),
                                 validation=train_val_test_dict.get('validation', None),
                                 test=train_val_test_dict.get('test', None),
                                 target_dir=target, action=action, delete_source=False,
                                 verbose=verbose)
    return output_dict


def image_divvy(instance,
                divvy_rule,
                db_to_extract='records_db',
                action='copy',
                train_val_test_dict=None,
                create_dirs=True,
                allow_overwrite=True,
                image_column=None,
                verbose=True):
    """

    Copy images from the cache to another location.

    :param instance: the yield of the yield of ``biovida.unification.unify_against_images()`` or an instance of
                     ``OpeniInterface`` or ``CancerImageInterface``.
    :type instance: ``OpeniInterface`` or ``CancerImageInterface`` or ``Pandas DataFrame``
    :param divvy_rule: must be a `function`` which (1) accepts a single parameter (argument) and (2) return
                       system path(s) [see example below].
    :type divvy_rule: ``function``
    :param db_to_extract: the database to use. Must be one of:

                    - 'records_db': the dataframe resulting from the most recent ``search()`` & ``pull()``.
                    - 'cache_records_db': the cache dataframe for ``instance``.
                    - 'unify_against_images': the yield of ``biovida.unification.unify_against_images()``.

    :type db_to_extract: ``str``
    :param action: one of: 'copy', 'ndarray'.

                    - if ``'copy'``: copy from files from the cache to (i) the location prescribed by ``divvy_rule``,
                      when ``train_val_test_dict=None``, else (ii) the 'target_location' key in ``train_val_test_dict``.

                    - if ``'ndarray'``: return a nested dictionary of ``ndarray`` ('numpy') arrays.

    :type action: ``str``
    :param train_val_test_dict: a dictionary denoting the proportions for any of: ``'train'``, ``'validation'`` and/or ``'test'``.

                        .. note:

                            If ``action='copy'``, a ``'target_dir'`` key (target directory) must also be included.

    :type train_val_test_dict: ``None`` or ``dict``
    :param create_dirs: if ``True``, create directories returned by ``divvy_rule`` if they do not exist. Defaults to ``True``.
    :type create_dirs: ``bool``
    :param allow_overwrite: if ``True`` allow existing images to be overwritten. Defaults to ``True``.
    :type allow_overwrite: ``bool``
    :param image_column: the column to use when copying images. If ``None``, use ``'cached_images_path'``. Default to ``None``.
    :type image_column: ``str``
    :param verbose: if ``True`` print additional details. Defaults to ``True``.
    :type verbose: ``bool``

    :Example:

    >>> from biovida.images import image_divvy
    >>> from biovida.images import OpeniInterface

    |
    | **Obtain Images**

    >>> opi = OpeniInterface()
    >>> opi.search(image_type=['mri', 'ct'])
    >>> opi.pull()

    |
    | **Usage 1**: Copy Images from the Cache to a New Location

    >>> image_divvy(opi, divvy_rule="/your/output/path/here/output")

    |
    | **Usage 2a**: A Rule which Invariably Returns a Single Save Location for a Single Row

    >>> def my_divvy_rule1(row):
    >>>     if isinstance(row['image_modality_major'], str):
    >>>         if 'mri' == row['image_modality_major']:
    >>>             return '/your/path/here/MRI_images'
    >>>         elif 'ct' == row['image_modality_major']:
    >>>             return '/your/path/here/CT_images'
    ...
    >>> image_divvy(opi, divvy_rule=my_divvy_rule1)

    |
    | **Usage 2b**: A Rule which can Return Multiple Save Locations for a Single Row

    >>> def my_divvy_rule2(row):
    >>>     locations = list()
    >>>     if isinstance(row['image_modality_major'], str):
    >>>         if 'leg' in row['abstract']:
    >>>             locations.append('/your/path/here/leg_images')
    >>>         if 'pelvis' in row['abstract']:
    >>>             locations.append('/your/path/here/pelvis_images')
    >>>     return locations
    ...
    >>> image_divvy(opi, divvy_rule=my_divvy_rule2)

    |
    | **Usage 3a**: Divvying into *train*/*validation*/*test*

    >>> def my_divvy_rule3(row):
    >>>     if isinstance(row['image_modality_major'], str):
    >>>         if 'mri' == row['image_modality_major']:
    >>>             return 'mri'
    >>>         elif 'ct' == row['image_modality_major']:
    >>>             return 'ct'

    Copying to a New Location

    >>> train_val_test_dict = {'train': 0.7, 'test': 0.3, 'target_dir': '/your/path/here/output'}
    >>> image_divvy(opi, divvy_rule=my_divvy_rule3, action='copy', train_val_test_dict=train_val_test_dict)

    Obtaining ndarrays (numpy arrays)

    >>> train_val_test_dict = {'train': 0.7, 'validation': 0.2, 'test': 0.1, 'target_dir': '/your/path/here/output'}
    >>> tvt = image_divvy(opi, divvy_rule=my_divvy_rule3, action='ndarray', train_val_test_dict=train_val_test_dict)

    The resultant ndarrays can be unpacked into objects as follows:

    >>> train_ct, train_mri = tvt['train']['ct'], tvt['train']['mri']
    >>> val_ct, val_mri = tvt['validation']['ct'], tvt['validation']['mri']
    >>> test_ct, test_mri = tvt['test']['ct'], tvt['test']['mri']

    .. note::

        Divvying into *train*/*validation*/*test* is powered by the ``train_val_test`` function
        (available :func:`here <biovida.support_tools.utilities.train_val_test>`).

    .. note::

        If a function passed to ``divvy_rule`` returns a system path when a dictionary has been passed
        to ``train_val_test_dict``, only the basename of the system path will be used.

    .. warning::

        While it is possible to pass a function to ``divvy_rule`` which returns multiple categories
        (similar to ``my_divvy_rule2()``) when divvying into *train*/*validation*/*test*, doing
        so is not recommended. Overlap between these groups is likely to lead to erroneous
        performance metrics (e.g., accuracy) when assessing fitted models.

    """
    _image_divvy_error_checking(action=action, train_val_test_dict=train_val_test_dict)

    data_frame = getattr(instance, db_to_extract) if db_to_extract != 'unify_against_images' else instance
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("{0} expected to be a DataFrame.\n"
                        "Got an object of type: '{1}'.".format(db_to_extract, type(data_frame).__name__))

    # Define the column to copy images from.
    column_to_use = _divvy_column_selector(instance, db_to_extract, image_column, data_frame)

    divvy_rule_wrapper = _image_divvy_wrappers_gen(divvy_rule=divvy_rule, action=action,
                                                   train_val_test_dict=train_val_test_dict,
                                                   column_to_use=column_to_use, create_dirs=create_dirs,
                                                   allow_overwrite=allow_overwrite)

    def divvy_rule_apply():
        divvy_info = list()
        for _, row in data_frame.iterrows():
            target = divvy_rule_wrapper(row)
            if isinstance(target, list):
                for t in target:
                    divvy_info.append(t)
        return pd.DataFrame(divvy_info) if len(divvy_info) else None

    divvy_info = divvy_rule_apply()

    if isinstance(divvy_info, pd.DataFrame):
        return _image_divvy_train_val_test_wrapper(action=action, verbose=verbose, divvy_info=divvy_info,
                                                   train_val_test_dict=train_val_test_dict)
    else:
        return None























