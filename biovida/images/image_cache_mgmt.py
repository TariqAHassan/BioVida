# coding: utf-8

"""

    Image Cache Management
    ~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import shutil
import numpy as np
import pandas as pd
from warnings import warn
from collections import Counter
from collections import defaultdict

# General Support Tools
from biovida.support_tools.support_tools import (tqdm,
                                                 cln,
                                                 multimap,
                                                 directory_existence_handler,
                                                 ipython_display,
                                                 IN_NOTEBOOK)

# Utilities
from biovida.support_tools.utilities import (_tvt_dict_gen,
                                             train_val_test,
                                             _train_val_test_engine,
                                             _file_paths_dict_to_ndarrays)

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
        return {k: tuple(sorted(v)) if isinstance(v, (list, tuple)) else v for k, v in d.items()}

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
    try:
        joined_db = records_db.join(update_db, how='left')
    except:
        joined_db = records_db
        for c in update_db.columns:
            joined_db[c] = update_db[c]

    return joined_db.fillna(np.NaN).dropna(subset=list(update_db.columns), how='all').reset_index(drop=True)


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
                      duplicates,
                      rows_to_conserve_func=None,
                      pre_return_func=None,
                      columns_with_iterables_to_sort=None):
    """

    Merge the existing cache record database with new additions.

    .. warning::

        Both ``current_records_db`` and ``records_db_update`` are
        expected to have 'biovida_version' and 'pull_time' columns.

    :param current_records_db: the existing record database.
    :type current_records_db: ``Pandas DataFrame``
    :param records_db_update: the new records dataframe to be merged with the existing one (``current_records_db``).
    :type records_db_update: ``Pandas DataFrame``
    :param columns_with_dicts: a list of columns which contain dictionaries. Note: this column *should* contain only
                               dictionaries or NaNs.
    :type columns_with_dicts: ``list``, ``tuple`` or ``None``
    :param duplicates: A function to handle dropping duplicates. This function should accept a dataframe
                      (and *only* a dataframe) and return a dataframe.
    :type duplicates: ``func``
    :param rows_to_conserve_func: function to generate a list of booleans which denote whether or not the image is,
                                  in fact, present in the cache. If not, remove it from the database to be saved.
    :type rows_to_conserve_func: ``func``
    :param pre_return_func: a function to apply to the dataframe before it is returned. This function should accept a
                            dataframe (and *only* a dataframe) and return a dataframe.
    :type pre_return_func: ``None`` or ``func``
    :param columns_with_iterables_to_sort: columns which themselves contain lists or tuples which should be sorted
                                           prior to dropping. Defaults to ``None``.
    :type columns_with_iterables_to_sort: ``list`` or ``tuple``
    :return: a dataframe which merges ``current_records_db`` and ``records_db_update``
    :rtype: ``Pandas DataFrame``
    """
    # Note: this function does not explicitly handle cases where
    #       combined_dbs has length 0, no obvious need to though.
    combined_dbs = pd.concat([current_records_db, records_db_update], ignore_index=True)

    # Mark each row to conserve order following ``pandas.drop_duplicates()``.
    combined_dbs['__temp_order__'] = range(combined_dbs.shape[0])

    if callable(rows_to_conserve_func):
        combined_dbs = combined_dbs[combined_dbs.apply(rows_to_conserve_func, axis=1)]

    # Convert items in ``columns_with_dicts`` from dictionaries to tuple of tuples.
    # (making them hashable, as required by ``pandas.drop_duplicates()``).
    combined_dbs = multimap(combined_dbs, columns=columns_with_dicts, func=_dict_to_tot)

    combined_dbs = multimap(combined_dbs, columns=columns_with_iterables_to_sort,
                            func=lambda x: tuple(sorted(x)))

    combined_dbs = duplicates(combined_dbs)
    combined_dbs = multimap(combined_dbs, columns=columns_with_dicts, func=dict)
    combined_dbs = combined_dbs.sort_values('__temp_order__')
    combined_dbs = _relationship_mapper(data_frame=combined_dbs, interface_name=interface_name)

    if callable(pre_return_func):
        combined_dbs = pre_return_func(combined_dbs)

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
    'OpeniImageProcessing': ('cached_images_path',),
    'unify_against_images': ('cached_images_path',),
    'CancerImageInterface': ('cached_dicom_images_path', 'cached_images_path')
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
        raise ValueError("Action Canceled. 'y' must be entered to proceed.")


def _most_recent_pull_time(instance):
    """
    
    Extract the time of the most recent 
    data pull from ``instance``.
    
    :param instance: see ``image_delete()``
    :type instance: see ``image_delete()``
    :return: see description.
    :rtype: ``bool``
    """
    if instance.__class__.__name__ == 'OpeniImageProcessing':
        records_db = instance.instance.records_db
    else:
        records_db = instance.records_db

    if isinstance(records_db, pd.DataFrame):
        pull_time = records_db['pull_time'].iloc[0]
    else:
        raise TypeError("`only_recent=True` is invalid. The `records_db` "
                        "associated with `instance` is not a DataFrame.")

    return pull_time


def _pretty_print_image_delete(deleted_rows, verbose):
    """

    Pretty print ``deleted_rows``.

    :param deleted_rows: as evolved inside ``image_delete``.
    :type deleted_rows: ``dict``
    :param verbose: see ``image_delete``.
    :type verbose: ``bool``
    """
    if deleted_rows and verbose:
        print("\nIndices of Deleted Rows:\n")
        # Note: `.T` handles when values are of unequal length.
        deleted_rows_df = pd.DataFrame.from_dict(deleted_rows, orient='index').T
        # Put `records_db` before of `cache_records_db` if both are present.
        to_print = deleted_rows_df[sorted(deleted_rows_df.columns, reverse=True)]
        if len(to_print):
            if IN_NOTEBOOK:
                ipython_display(to_print)
            else:
                pandas_pprint(to_print, full_rows=True, suppress_index=True)
        else:
            print("\nNo Rows Deleted.")


def _delete_rule_wrapper_gen(instance, delete_rule, only_recent, image_columns):
    """
    
    Generate ``delete_rule_wrapper()``.
    
    :param instance: see ``image_delete()``
    :type instance: ``OpeniInterface``, ``OpeniImageProcessing`` or ``CancerImageInterface``
    :param delete_rule: see ``image_delete()``
    :type delete_rule: ``str`` or ``func``
    :param only_recent: see ``image_delete()``
    :type only_recent: ``bool``
    :param image_columns: as evolved in ``image_delete()``
    :type image_columns: ``tuple``
    :return: a function which wraps ``delete_rule`` in such a way that a
             boolean is always the output. 
    :rtype: ``func``
    """
    if isinstance(delete_rule, str):
        if cln(delete_rule).lower() == 'all':
            delete_all = True
        else:
            raise ValueError("`delete_rule` must be 'all' or of type 'function'.")
    else:
        delete_all = False

    last_pull_time = _most_recent_pull_time(instance) if only_recent else None

    def delete_rule_wrapper(row, enact):
        """Wrap delete_rule to ensure the output,
         whether or not to delete, is a boolean."""
        if callable(delete_rule):
            if only_recent:
                func_delete = delete_rule(row) and row['pull_time'] == last_pull_time
            else:
                func_delete = delete_rule(row)
        else:
            func_delete = False

        if only_recent:
            blanket_delete = delete_all and row['pull_time'] == last_pull_time
        else:
            blanket_delete = delete_all

        if blanket_delete or (isinstance(func_delete, bool) and func_delete is True):
            if enact:
                for c in image_columns:
                    _robust_delete(row[c])
            return False  # drop row from the dataframe
        else:
            return True   # keep row in the dataframe

    return delete_rule_wrapper


def image_delete(instance, delete_rule, only_recent=False, verbose=True):
    """

    Delete rows, and associated images, from ``records_db`` and ``cache_records_db``
    DataFrames associated with ``instance``.

    .. warning::

        The effects of this function can only be undone by downloading the deleted data again.
        
    .. warning::
    
        The default behavior for this function is to delete *all* rows (and associated image(s))
        that ``delete_rule`` authorizes. To limit deletion to the most recent data pull,
        ``only_recent`` must be set to ``True``.

    :param instance: an instance of ``OpeniInterface``, ``OpeniImageProcessing`` or ``CancerImageInterface``.
    :type instance: ``OpeniInterface``, ``OpeniImageProcessing`` or ``CancerImageInterface``
    :param delete_rule: must be one of: ``'all'`` (delete *all* data) or a ``function`` which (1) accepts a single
                        parameter (argument) and (2) returns ``True`` when the data is to be deleted.
    :type delete_rule: ``str`` or ``func``
    :param only_recent: if ``True``, only apply ``delete_rule`` to data obtained in the most recent pull. Defaults to ``False``.
    :type only_recent: ``bool``
    :param verbose: if ``True``, print additional information.
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
    # ToDo: refactor. This function is very cumbersome.
    index_dict = dict()
    _double_check_with_user()

    if verbose:
        print("\nDeleting...")

    image_columns = _image_instance_image_columns[instance.__class__.__name__]
    delete_rule_wrapper = _delete_rule_wrapper_gen(instance=instance, delete_rule=delete_rule,
                                                   only_recent=only_recent, image_columns=image_columns)

    def index_dict_update(data_frame_name, stage, data_frame):
        if stage == 'before':
            index_dict[data_frame_name] = {stage: data_frame.index.tolist()}
        elif stage == 'after':
            index_dict[data_frame_name][stage] = data_frame.index.tolist()

    def non_cache_db(data_frame_name, data_frame, enact):
        """Handle `records_db` and `image_dataframe`."""
        index_dict_update(data_frame_name, stage='before', data_frame=data_frame)
        to_conserve = data_frame.apply(lambda r: delete_rule_wrapper(r, enact=enact), axis=1).tolist()
        data_frame = data_frame[to_conserve].reset_index(drop=True)
        index_dict_update(data_frame_name, stage='after', data_frame=data_frame)
        return data_frame, to_conserve

    if instance.__class__.__name__ == 'OpeniImageProcessing':
        instance.image_dataframe, to_conserve = non_cache_db('image_dataframe', instance.image_dataframe, enact=True)
        if instance.db_to_extract == 'records_db':
            instance.instance.records_db = instance.instance.records_db[to_conserve].reset_index(drop=True)
        elif isinstance(instance.instance.records_db, pd.DataFrame):
            index_dict_update('records_db', 'before', data_frame=instance.instance.records_db)
            to_conserve = instance.instance.records_db[image_columns[0]].map(
                lambda x: os.path.isfile(x) if isinstance(x, str) else False)
            instance.instance.records_db = instance.instance.records_db[to_conserve].reset_index(drop=True)
            index_dict_update('records_db', 'after', data_frame=instance.instance.records_db)
        if isinstance(instance.instance.cache_records_db, pd.DataFrame):
            index_dict_update('cache_records_db', 'before', data_frame=instance.instance.cache_records_db)
            instance.instance._load_prune_cache_records_db(load=False)
            index_dict_update('cache_records_db', 'after', data_frame=instance.instance.cache_records_db)
        else:
            raise TypeError("`cache_record_db` is not a DataFrame.")
    else:
        if isinstance(instance.records_db, pd.DataFrame):
            instance.records_db, _ = non_cache_db('records_db', instance.records_db, enact=False)
        if isinstance(instance.cache_records_db, pd.DataFrame):
            index_dict_update('cache_records_db', 'before', data_frame=instance.cache_records_db)
            _ = instance.cache_records_db.apply(lambda r: delete_rule_wrapper(r, enact=True), axis=1)
            instance._load_prune_cache_records_db(load=False)
            instance.cache_records_db = _relationship_mapper(instance.cache_records_db, instance.__class__.__name__)
            instance._save_cache_records_db()
            index_dict_update('cache_records_db', 'after', data_frame=instance.cache_records_db)
        else:
            raise TypeError("`cache_record_db` is not a DataFrame.")

    deleted_rows = {k: sorted(set(v['before']) - set(v['after'])) for k, v in index_dict.items()}
    _pretty_print_image_delete(deleted_rows=deleted_rows, verbose=verbose)

    return deleted_rows


# ----------------------------------------------------------------------------------------------------------
# Divvy Image Data
# ----------------------------------------------------------------------------------------------------------


def _image_divvy_error_checking(divvy_rule, action, train_val_test_dict):
    """

    Check for possible errors for ``image_divvy()``.

    :param divvy_rule: see ``image_divvy()``.
    :type divvy_rule: ``str`` or ``func``
    :param train_val_test_dict: see ``image_divvy()``.
    :type action: ``str``
    :param train_val_test_dict: see ``image_divvy()``.
    :type train_val_test_dict: ``dict``
    """
    if not isinstance(divvy_rule, str) and not callable(divvy_rule):
        raise TypeError("`divvy_rule` must be a string or function.")

    if isinstance(train_val_test_dict, dict):
        if not train_val_test_dict:
            raise KeyError("`train_val_test_dict` is empty")
        if action == 'copy' and 'target_dir' not in train_val_test_dict:
            raise KeyError("`train_val_test_dict` must contain a `target_dir` key "
                           "if `action='copy'`.")
        for k in train_val_test_dict:
            if k not in ('train', 'validation', 'test', 'target_dir', 'delete_source'):
                raise KeyError("Invalid `train_val_test_dict` key: '{0}'.")
        if action == 'ndarray' and 'target_dir' in train_val_test_dict:
            warn("\nThe 'target_dir' entry in `train_val_test_dict` has no\n"
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
    _ = directory_existence_handler(path_=copy_to, allow_creation=allow_creation, verbose=True)

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


def _divvy_column_selector(instance, image_column, data_frame):
    """

    Select the column to use when copying images from.

    :param instance:  see ``image_divvy()``
    :type instance: ``OpeniInterface`` or ``CancerImageInterface``
    :param image_column: see ``image_divvy()``
    :type image_column: ``str``
    :param data_frame: as evolved inside  ``image_divvy()``.
    :type data_frame: ``Pandas DataFrame``
    :return: the column in ``data_frame`` to use when copying images to the new location.
    :rtype: ``str``
    """
    if image_column is None:
        return _image_instance_image_columns[type(instance).__name__][0]
    elif not isinstance(image_column, str):
        raise TypeError('`image_column` must be a string or `None`.')
    elif image_column in _image_instance_image_columns[instance.__class__.__name__]:
        if image_column in data_frame.columns:
            return image_column
        else:
            raise KeyError("The '{0}' column is missing from the dataframe.".format(image_column))
    else:
        raise KeyError("'{0}' is not a valid image column.".format(image_column))


def _image_divvy_wrappers_gen(divvy_rule, action, train_val_test_dict, column_to_use, create_dirs, allow_overwrite):
    """

    Wrap the ``divvy_rule`` passed to ``image_divvy()``.

    :param divvy_rule: see ``image_divvy()``.
    :type divvy_rule: ``str`` or ``func``
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
    :rtype: ``func``
    """
    def copy_rule_wrapper(row, copy_to):
        if isinstance(copy_to, (str, tuple, list)):
            all_copy_targets = [copy_to] if isinstance(copy_to, str) else copy_to
            if not len(all_copy_targets):
                return None
            for i in all_copy_targets:
                if not isinstance(i, str):
                    raise TypeError("`divvy_rule` returned an iterable containing "
                                    "a element which is not a string.")
                _robust_copy(to_copy=row[column_to_use], copy_to=i,
                             allow_creation=create_dirs, allow_overwrite=allow_overwrite)
        elif copy_to is not None:
            raise TypeError("String, list or tuple expected. "
                            "`divvy_rule` returned an object of type "
                            "'{0}'.".format(type(copy_to).__name__))

    def divvy_rule_wrapper(row):
        group = divvy_rule(row) if callable(divvy_rule) else divvy_rule
        if not isinstance(group, (str, list, tuple)):
            return None
        if isinstance(group, (list, tuple)) and not len(group):
            return None
        if action == 'copy' and not isinstance(train_val_test_dict, dict):
            copy_rule_wrapper(row, group)
        if isinstance(row[column_to_use], (str, tuple, list)):
            cache_info = [row[column_to_use]] if isinstance(row[column_to_use], str) else list(row[column_to_use])
            if isinstance(group, str):
                return [[os.path.basename(group), cache_info]]
            elif isinstance(group, (list, tuple)):
                return [[os.path.basename(c), cache_info] for c in group]
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
    :param divvy_info: as evolved inside ``image_divvy`` (by ``divvy_info_data_frame_to_dict()``).
    :type divvy_info: ``dict``
    :param train_val_test_dict see ``image_divvy()``.
    :type train_val_test_dict:
    :return: see ``train_val_test``.
    :rtype: ``dict``
    """
    random_state = train_val_test_dict.get('random_state', None)
    target = train_val_test_dict.get('target_dir') if action == 'copy' else None
    delete_source = train_val_test_dict.get('delete_source', False) if action == 'copy' else False

    output_dict = train_val_test(data=divvy_info,
                                 train=train_val_test_dict.get('train', None),
                                 validation=train_val_test_dict.get('validation', None),
                                 test=train_val_test_dict.get('test', None),
                                 target_dir=target, action=action, delete_source=delete_source,
                                 random_state=random_state, verbose=verbose)
    return output_dict


def _divvy_info_to_dict(divvy_info):
    """

    Convert the list evolved inside ``divvy_rule_apply()`` in
    ``image_divvy()`` into a dictionary.

    :param divvy_info: a list of the form ``[[string_1, ['a', 'b']], [string_1, ['c', 'd']], [string_2, ['e']], ...]``.
    :type divvy_info: ``list``
    :return: a dictionary form: ``{string_1: ['a', 'b', 'c', 'd'], string_2: ['e'], ...}``.
    :rtype: ``dict``
    """
    d = defaultdict(list)
    for (k, v) in divvy_info:
        d[k] += v
    return dict(d)


def _divvy_openi_image_processing(instance,
                                  divvy_rule,
                                  action,
                                  train_val_test_dict,
                                  create_dirs,
                                  allow_overwrite,
                                  verbose):
    """

    Handle the special case of 'divvying' when an instance of the ``OpeniImageProcessing``
    class is passed to ``image_divvy()``.

    This has to be to be handled separately so that cropping information,
    determined by ``OpeniImageProcessing``'s analysis methods, can be applied.

    :param instance: see ``image_divvy()``.
    :type instance: ``OpeniImageProcessing``
    :param divvy_rule:  see ``image_divvy()``.
    :type divvy_rule: ``str`` or ``func``
    :param action:  see ``image_divvy()``.
    :type action: ``str``
    :param train_val_test_dict: see ``image_divvy()``.
    :type train_val_test_dict: ``dict`` or ``None``
    :param create_dirs:  see ``image_divvy()``.
    :type create_dirs: ``bool``
    :param allow_overwrite: see ``image_divvy()``.
    :type allow_overwrite: ``bool``
    :param verbose: see ``image_divvy()``.
    :type verbose: ``bool``
    :return: yield of ``OpeniImageProcessing.output`` or ``_train_val_test_engine()``.
    :rtype: ``dict``
    """
    if action in ('copy', 'ndarray') and not isinstance(train_val_test_dict, dict):
        output_dict = instance.output(output_rule=divvy_rule, create_dirs=create_dirs,
                                      allow_overwrite=allow_overwrite, action=action)
    elif action in ('copy', 'ndarray') and isinstance(train_val_test_dict, dict):
        if action == 'copy':
            ndarray_with_path = True
            engine_action = 'write_ndarray'
            target_path = train_val_test_dict['target_dir']
        elif action == 'ndarray':
            ndarray_with_path = False
            engine_action = None
            target_path = None

        divvy_info = instance.output(output_rule=divvy_rule,
                                     action='ndarray',  # used so that cropping is applied.
                                     ndarray_with_path=ndarray_with_path)

        output_dict = _train_val_test_engine(action=engine_action,
                                             tvt=_tvt_dict_gen(train_val_test_dict),
                                             group_files_dict=divvy_info,
                                             target_path=target_path,
                                             random_state=train_val_test_dict.get('random_state', None),
                                             allowed_actions=('write_ndarray', None),
                                             verbose=verbose)

    return output_dict


def image_divvy(instance,
                divvy_rule,
                action='ndarray',
                db_to_extract='records_db',
                train_val_test_dict=None,
                create_dirs=True,
                allow_overwrite=True,
                image_column=None,
                stack=True,
                verbose=True):
    """

    Grouping Cached Images.

    .. warning::

        Currently, if an ``OpeniImageProcessing`` instance is first passed to
        ``biovida.unification.unify_against_images`` and then to this function,
        image cropping will not be applied.

    :param instance: the yield of the yield of ``biovida.unification.unify_against_images()`` or an instance of
                     ``OpeniInterface``, ``OpeniImageProcessing`` or ``CancerImageInterface``.
    :type instance: ``OpeniInterface``, ``OpeniImageProcessing``, ``CancerImageInterface`` or ``Pandas DataFrame``
    :param divvy_rule: must be a `function`` which (1) accepts a single parameter (argument) and (2) return
                       system path(s) [see example below].
    :type divvy_rule: ``str`` or ``func``
    :param action: one of: ``'copy'``, ``'ndarray'``.

                    - if ``'copy'``: copy from files from the cache to (i) the location prescribed by ``divvy_rule``,
                      when ``train_val_test_dict=None``, else (ii) the 'target_location' key in ``train_val_test_dict``.

                    - if ``'ndarray'``: return a nested dictionary of ``ndarray`` ('numpy') arrays (default).

    :type action: ``str``
    :param db_to_extract: the database to use. Must be one of:

        * 'records_db': the dataframe resulting from the most recent ``search()`` & ``pull()`` (default).
        * 'cache_records_db': the cache dataframe for ``instance``.
        * 'unify_against_images': the yield of ``biovida.unification.unify_against_images()``.

        .. note::

            If an instance of ``OpeniImageProcessing`` is passed, the dataframe will be extracted automatically.

    :type db_to_extract: ``str``
    :param train_val_test_dict: a dictionary denoting the proportions for any of: ``'train'``, ``'validation'`` and/or ``'test'``.

                        .. note::

                            * If ``action='copy'``, a ``'target_dir'`` key (target directory) *must* also be included.
                            * A ``'random_state'`` key can be passed, with an integer as the value, to seed shuffling.
                            * To delete the source files, a ``'delete_source'`` key may be included (optional).
                              The corresponding value provided *must* be a boolean. If no such key key is provided,
                              ``'delete_source'`` defaults to ``False``.

    :type train_val_test_dict: ``None`` or ``dict``
    :param create_dirs: if ``True``, create directories returned by ``divvy_rule`` if they do not exist. Defaults to ``True``.
    :type create_dirs: ``bool``
    :param allow_overwrite: if ``True`` allow existing images to be overwritten. Defaults to ``True``.
    :type allow_overwrite: ``bool``
    :param image_column: the column to use when copying images. If ``None``, use ``'cached_images_path'``. Defaults to ``None``.
    :type image_column: ``str``
    :param stack: if ``True``, stack 3D volumes and time-series images when ``action='ndarray'``. Defaults to ``True``.
    :type stack: ``bool``
    :param verbose: if ``True`` print additional details. Defaults to ``True``.
    :type verbose: ``bool``
    :return:

        * If ``divvy_rule`` is a string:

          * If ``action='copy'`` and ``train_val_test_dict`` is not dictionary, this function will
            return a dictionary of the form ``{divvy_rule: [cache_file_path, cache_file_path, ...], ...}``.

        * If ``divvy_rule`` is a function:

          * If ``action='copy'`` and ``train_val_test_dict`` is not a dictionary, this function will
            return a dictionary of the form ``{string returned by divvy_rule(): [cache_file_path, cache_file_path, ...], ...}``.

          * If ``action='ndarray'`` and ``train_val_test_dict`` is not a dictionary, this function will
            return a dictionary of the form ``{string returned by divvy_rule(): array([Image Matrix, Image Matrix, ...]), ...}``.

          * If ``train_val_test_dict`` is a dictionary, the output is powered by utilities.train_val_test
            (available :func:`here <biovida.support_tools.utilities.train_val_test>`).

    :rtype: ``dict``

    :Example:

    >>> from biovida.images import image_divvy
    >>> from biovida.images import OpeniInterface

    |
    | **Obtain Images**

    >>> opi = OpeniInterface()
    >>> opi.search(image_type=['mri', 'ct'])
    >>> opi.pull()

    |
    | **Usage 1a**: Copy Images from the Cache to a New Location

    >>> summary_dict = image_divvy(opi, divvy_rule="/your/output/path/here/output", action='copy')

    |
    | **Usage 1b**: Converting to ``ndarrays``

    >>> def my_divvy_rule1(row):
    >>>     if isinstance(row['image_modality_major'], str):
    >>>         if 'mri' == row['image_modality_major']:
    >>>             return 'mri'
    >>>         elif 'ct' == row['image_modality_major']:
    >>>             return 'ct'
    ...
    >>> nd_data = image_divvy(opi, divvy_rule=my_divvy_rule1, action='ndarray')

    The resultant ``ndarrays`` can be extracted as follows:

    >>> ct_images = nd_data['ct']
    >>> mri_images = nd_data['mri']

    |
    | **Usage 2a**: A Rule which Invariably Returns a Single Save Location for a Single Row

    >>> def my_divvy_rule2(row):
    >>>     if isinstance(row['image_modality_major'], str):
    >>>         if 'mri' == row['image_modality_major']:
    >>>             return '/your/path/here/MRI_images'
    >>>         elif 'ct' == row['image_modality_major']:
    >>>             return '/your/path/here/CT_images'
    ...
    >>> summary_dict = image_divvy(opi, divvy_rule=my_divvy_rule2, action='copy')

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
    >>> summary_dict= image_divvy(opi, divvy_rule=my_divvy_rule2, action='copy')

    |
    | **Usage 3**: Divvying into *train/validation/test*

    **i**. Copying to a New Location (reusing ``my_divvy_rule1``)

    >>> train_val_test_dict = {'train': 0.7, 'test': 0.3, 'target_dir': '/your/path/here/output'}
    >>> summary_dict = image_divvy(opi, divvy_rule=my_divvy_rule1, action='copy', train_val_test_dict=train_val_test_dict)

    **ii**. Obtaining ``ndarrays`` (numpy arrays)

    >>> train_val_test_dict = {'train': 0.7, 'validation': 0.2, 'test': 0.1}
    >>> image_dict = image_divvy(opi, divvy_rule=my_divvy_rule1, action='ndarray', train_val_test_dict=train_val_test_dict)

    The resultant ``ndarrays`` can be unpacked as follows:

    >>> train_ct, train_mri = image_dict['train']['ct'], image_dict['train']['mri']
    >>> val_ct, val_mri = image_dict['validation']['ct'], image_dict['validation']['mri']
    >>> test_ct, test_mri = image_dict['test']['ct'], image_dict['test']['mri']

    This function behaves the same if passed an instance of ``OpeniImageProcessing``

    >>> from biovida.images import OpeniImageProcessing
    >>> ip = OpeniImageProcessing(opi)
    >>> ip.auto()
    >>> ip.clean_image_dataframe()

    >>> image_dict = image_divvy(ip, divvy_rule=my_divvy_rule1, action='ndarray', train_val_test_dict=train_val_test_dict)
    ...

    .. note::

        If an instance of ``OpeniImageProcessing`` is passed to ``image_divvy``, the ``image_data_frame_cleaned``
        dataframe will be extracted.

    .. note::

        If a function passed to ``divvy_rule`` returns a system path when a dictionary has been passed
        to ``train_val_test_dict``, only the basename of the system path will be used.

    .. warning::

        While it is possible to pass a function to ``divvy_rule`` which returns multiple categories
        (similar to ``my_divvy_rule2()``) when divvying into *train/validation/test*, doing
        so is not recommended. Overlap between these groups is likely to lead to erroneous
        performance metrics (e.g., accuracy) when assessing fitted models.

    """
    # ToDo: remove need for the first (`OpeniImageProcessing`-related) warning given in the docstring above.
    _image_divvy_error_checking(divvy_rule=divvy_rule, action=action,
                                train_val_test_dict=train_val_test_dict)

    if type(instance).__name__ == 'OpeniImageProcessing':
        return _divvy_openi_image_processing(instance=instance, divvy_rule=divvy_rule,
                                             action=action, train_val_test_dict=train_val_test_dict,
                                             create_dirs=create_dirs, allow_overwrite=allow_overwrite,
                                             verbose=verbose)
    elif db_to_extract == 'unify_against_images':
        data_frame = instance
    else:
        data_frame = getattr(instance, db_to_extract)

    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("Expected a DataFrame.\n"
                        "Got an object of type: '{0}'."
                        "`db_to_extract` may be invalid".format(type(data_frame).__name__))

    column_to_use = _divvy_column_selector(instance, image_column=image_column, data_frame=data_frame)

    divvy_rule_wrapper = _image_divvy_wrappers_gen(divvy_rule=divvy_rule, action=action,
                                                   train_val_test_dict=train_val_test_dict,
                                                   column_to_use=column_to_use, create_dirs=create_dirs,
                                                   allow_overwrite=allow_overwrite)

    def divvy_rule_apply():
        divvy_info = list()
        for _, row in tqdm(data_frame.iterrows(), total=len(data_frame),
                           desc='Applying Divvy Rule', disable=not verbose):
            target = divvy_rule_wrapper(row)
            if isinstance(target, list):
                for t in target:
                    divvy_info.append(t)
        return _divvy_info_to_dict(divvy_info) if len(divvy_info) else None

    divvy_info = divvy_rule_apply()

    if isinstance(divvy_info, dict) and isinstance(train_val_test_dict, dict):
        return _image_divvy_train_val_test_wrapper(action=action, verbose=verbose, divvy_info=divvy_info,
                                                   train_val_test_dict=train_val_test_dict)
    elif isinstance(divvy_info, dict) and action == 'ndarray':
        return _file_paths_dict_to_ndarrays(divvy_info, dimensions=1, stack=stack, verbose=verbose)
    elif isinstance(divvy_info, dict) and action == 'copy':
        return divvy_info
