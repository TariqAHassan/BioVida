"""

    Image Cache Management
    ~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import os
import numpy as np
import pandas as pd
from warnings import warn
from collections import Counter

# General Support Tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import multimap

# General Image Support Tools
from biovida.images._image_tools import ActionVoid


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

    # Get duplicated img_large occurences. Use of 'img_large' is arbitrary, could have used
    # 'img_thumb', 'img_grid150', etc.
    duplicated_img_refs = (k for k, v in Counter(df['img_large']).items() if v > 1)

    # Get the indices of duplicates
    dup_index = {k: df[df['img_large'] == k].index.tolist() for k in duplicated_img_refs}

    def related(img_large, index):
        """Function to look for references to the same image in the cache."""
        if img_large in dup_index:
            return tuple(sorted([i for i in dup_index[img_large] if i != index]))
        else:
            return np.NaN

    # Apply `relate()`
    df['shared_image_ref'] = [related(img, index) for img, index in zip(df['img_large'], df.index)]

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


# ToDo: handle dataframes with 0 rows.
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
# Deleting Image Data
# ----------------------------------------------------------------------------------------------------------


_image_interface_image_columns = {
    'OpeniInterface': ('cached_images_path',),
    'CancerImageInterface': ('cached_images_path', 'cached_dicom_images_path')
}


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

    Ask the user to verify they wish to precede.

    """
    response = input("This action cannot be undone.\n"
                     "Are you sure you wish to precede (y/n)?")
    if cln(response).lower() not in ('yes', 'ye', 'es', 'y'):
        raise ActionVoid


def image_delete(interface, delete_rule):
    """

    .. warning::

        The effects of this function can only be undone by downloading the deleted data again.

    :param interface: an instance of ``OpeniInterface`` or ``CancerImageInterface``.
    :type interface: ``OpeniInterface`` or ``CancerImageInterface``
    :param delete_rule: must be one of: ``'all'`` (delete *all* data) or a ``function`` which (1) accepts a single
                        parameter (argument) and (2) returns ``True`` when the data is to be deleted.
    :type delete_rule: ``str`` or ``function``

    :Example:

    >>> from biovida.images import image_delete
    >>> from biovida.images import OpeniInterface
    ...
    >>> opi = OpeniInterface()
    >>> opi.search(image_type=['ct', 'mri'], collection='medpix')
    >>> opi.pull()
    ...
    >>> def my_delete_rule(row):
    >>>     if 'Oompa Loompas' in row['abstract']:
    >>>         return True
    ...
    >>> image_delete(opi, delete_rule=my_delete_rule)

    .. note::

        In this example, any rows in the ``records_db`` and ``cache_records_db``
        for which the 'abstract' column contains the string 'Oompa Loompas' will be deleted.
        Any images associated with this row will also be destroyed.

    """
    _double_check_with_user()

    delete_all = False
    if isinstance(delete_rule, str):
        if cln(delete_rule).lower() == 'all':
            delete_all = True
        else:
            raise ValueError("`delete_rule` must be 'all' or a `function`.")

    def delete_rule_wrapper(row, enact):
        """Wrap delete_rule to ensure the output is boolean"""
        if delete_all or delete_rule(row):
            if enact:
                for c in _image_interface_image_columns[interface.__class__.__name__]:
                    _robust_delete(row[c])
            return False
        else:
            return True

    if type(interface.records_db).__name__ == 'DataFrame':
        to_conserve = interface.records_db.apply(lambda r: delete_rule_wrapper(r, enact=False), axis=1)
        interface.records_db = interface.records_db[to_conserve.tolist()].reset_index(drop=True)
    if type(interface.cache_records_db).__name__ == 'DataFrame':
        # Apply ``delete_rule`` to ``cache_records_db``.
        interface.cache_records_db.apply(lambda r: delete_rule_wrapper(r, enact=True), axis=1)
        # Prune ``cache_records_db`` by inspecting which images have been deleted.
        interface._load_prune_cache_records_db(load=False)
        # Map relationships, if applicable.
        interface.cache_records_db = _relationship_mapper(interface.cache_records_db, interface.__class__.__name__)
        # Save the updated ``cache_records_db`` to 'disk'.
        interface._save_cache_records_db()
    else:
        raise TypeError("`cache_record_db` is not a DataFrame.")


























