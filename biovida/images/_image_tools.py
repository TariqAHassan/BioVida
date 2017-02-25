"""

    General Tools for the Image Subpackage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from time import sleep
from warnings import warn
from scipy.misc import imread, imresize
from skimage.color.colorconv import rgb2gray

from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import items_null


class NoResultsFound(Exception):
    pass


def sleep_with_noise(amount_of_time, mean=0.0, noise=0.75):
    """

    Sleep the current python instance by `amount_of_time`.

    :param amount_of_time: the amount of time to sleep the instance for.
    :type amount_of_time: ``int`` or ``float``
    :param mean: see ``numpy.random.normal()``. Defaults to 0.0.
    :type mean: ``float``
    :param noise: see ``numpy.random.normal()``. Defaults to 0.75.
    :type noise: ``float``
    """
    sleep(abs(amount_of_time + np.random.normal(loc=mean, scale=noise)))


def resetting_label(to_label):
    """

    Label repeats in a list.

    :param to_label: a list with repeating elements.
    :type to_label: ``tuple`` or ``list``
    :return: a list of string of the form shown in the example below.
    :rtype: ``list``


    :Example:

    >>> l = ['a', 'a', 'a', 'b', 'b', 'z']
     print(_reseting_label(to_label=l))
     ...
     ['a_1', 'a_2', 'a_3', 'b_1', 'b_2', 'z_1']
    """
    def formatted_label(existing_name, label):
        """Joins the items in `to_label` with the label number generated below."""
        head = cln(str(existing_name)) if not items_null(existing_name) and existing_name is not None else ""
        return "{0}_{1}".format(head, str(label))

    label = 1
    prior = to_label[0]
    all_labels = [formatted_label(prior, label)]
    for i in to_label[1:]:
        if i == prior:
            label += 1
            all_labels.append(formatted_label(prior, label))
        elif i != prior:
            label = 1
            prior = i
            all_labels.append(formatted_label(prior, label))

    return all_labels


def dict_to_tot(d):
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


def load_img_rescale(path_to_image, gray_only=False):
    """

    Loads an image, converts it to grayscale and normalizes (/255.0).

    :param path_to_image: the address of the image.
    :type path_to_image: ``str``
    :return: the image as a matrix.
    :rtype: ``ndarray``
    """
    if gray_only:
        return rgb2gray(path_to_image) / 255.0
    else:
        return rgb2gray(imread(path_to_image, flatten=True)) / 255.0


def image_transposer(converted_image, img_size, axes=(2, 0, 1)):
    """

    Tool to resize and transpose an image (about given axes).

    :param converted_image: the image as a ndarray.
    :type converted_image: ``ndarray``
    :param img_size: to size to coerse the images to be, e.g., (150, 150)
    :type img_size: ``tuple``
    :param axes: the axes to transpose the image on.
    :type axes: ``tuple``
    :return: the resized and transposed image.
    :rtype: ``ndarray``
    """
    return np.transpose(imresize(converted_image, img_size), axes).astype('float32')


def load_and_scale_imgs(list_of_images, img_size, axes=(2, 0, 1), status=True, grayscale_first=False):
    """

    Load and scale a list of images from a directory

    :param list_of_images: a list of paths to images.
    :type list_of_images: ``list`` or ``tuple``
    :param img_size: to size to coerse the images to be, e.g., (150, 150)
    :type img_size: ``tuple``
    :param axes: the axes to transpose the image on.
    :type axes: ``tuple``
    :param status: if ``True``, use `tqdm` to print progress as the load progresses.
    :type status: ``bool``
    :param grayscale_first: convert the image to grayscale first.
    :type grayscale_first: ``bool``
    :return: the images as ndarrays nested inside of another ndarray.
    :rtype: ``ndarray``
    """
    # Source: https://blog.rescale.com/neural-networks-using-keras-on-rescale/
    def status_bar(x):
        if status:
            return tqdm(x)
        else:
            return x

    def load_func(img):
        if 'ndarray' in str(type(img)):
            converted_image = img
        else:
            # Load grayscale images by first converting them to RGB (otherwise, `imresize()` will break).
            if grayscale_first:
                loaded_img = Image.open(img).convert("LA")
                loaded_img = loaded_img.convert("RGB")
            else:
                loaded_img = Image.open(img).convert("RGB")
            converted_image = np.asarray(loaded_img)
        return image_transposer(converted_image, img_size, axes=axes)

    return np.array([load_func(img_name) for img_name in status_bar(list_of_images)]) / 255.0


def show_plt(img):
    """

    Use matplotlib to display an img (which is represented as a matrix).

    :param img: an img represented as a matrix.
    :type img: ``ndarray``
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


# ----------------------------------------------------------------------------------------------------------
# Handling Image Databases
# ----------------------------------------------------------------------------------------------------------


def record_update_dbs_joiner(record_db, update_db):
    """

    Join and drop rows for which `update_db`'s columns exclusively contain NaNs.

    :param record_db: permanent database/dataframe which keeps a record of files in the cache.
    :type record_db: ``Pandas DataFrame``
    :param update_db: database/dataframe to 'update' ``record_db``
    :type update_db: ``Pandas DataFrame``
    :return: ``record_db`` with ``update_db`` left-joined.
    :rtype: ``Pandas DataFrame``
    """
    joined_db = record_db.join(update_db, how='left').dropna(subset=list(update_db.columns), how='all')
    return joined_db.reset_index(drop=True)


def load_temp_dbs(temp_db_path):
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
            warn("FileNotFoundError: Either '{0}__record_db.p' or '{0}__update_db.p'"
                 " is missing from\n {1}\nDeleting the file which is present...\n"
                 "As a result, images obtained from the last `pull()` will likely be missing"
                 " from `cache_record_db`.\nFor this reason, it is recommended that you\n"
                 "**precisely** repeat your last `search()` and `pull()`.".format(pull_time, temp_db_path))
        else:
            groupings.append(group)

    if not len(groupings):
        return None

    # Read the dataframes in the '__temp__' directory into memory
    frames = list()
    for group in groupings:
        record_db = pd.read_pickle([i for i in group if "__record_db.p" in i][0])
        update_db = pd.read_pickle([i for i in group if "__update_db.p" in i][0])
        frames.append(record_update_dbs_joiner(record_db, update_db))

    # Concatenate all frames
    return pd.concat(frames, ignore_index=True)


def record_db_merge(current_record_db,
                    record_db_update,
                    query_column_name,
                    pull_time_column_name,
                    duplicates_subset_columns,
                    rows_to_conserve_func=None,
                    post_concat_mapping=None,
                    columns_with_iterables_to_sort=None,
                    relationship_mapping_func=None):
    """

    Merge the existing record database with new additions.

    :param current_record_db: the existing record database.
    :type current_record_db: ``Pandas DataFrame``
    :param record_db_update: the new records dataframe to be merged with the existing one (``current_record_db``).
    :type record_db_update: ``Pandas DataFrame``
    :param query_column_name: the column which contains the query responcible for the results. Note: this column
                              *should* contain only dictionaries.
    :type query_column_name: ``str``
    :param pull_time_column_name: the name of the column with the time the query was created.
    :type pull_time_column_name: ``str``
    :param duplicates_subset_columns: a list (or tuple) of columns to consider when dropping duplicates.
    :type duplicates_subset_columns: ``list`` or ``tuple``
    :param rows_to_conserve_func: function to generate a list of booleans which denote whether or not the image is,
                                  in fact, present in the cahce. If not, remove it from the database to be saved.
    :type rows_to_conserve_func: ``function``
    :param post_concat_mapping: a list (or tuple) of the form (new column name, column to apply the func to, func).
    :type post_concat_mapping: ``list`` or ``tuple``
    :param columns_with_iterables_to_sort: columns which themselves contain lists or tuples which should be sorted
                                  prior to dropping. Defaults to ``None``.
    :type columns_with_iterables_to_sort: ``list`` or ``tuple``
    :param relationship_mapping_func: function to map relationships in the dataframe. Defaults to ``None``.
    :type relationship_mapping_func: ``function``
    :return: a dataframe which merges ``current_record_db`` and ``record_db_update``
    :rtype: ``Pandas DataFrame``
    """
    # Load in the current database and combine with the `record_db_update` database
    combined_dbs = pd.concat([current_record_db, record_db_update], ignore_index=True)

    # Mark each row to conserve order following ``pandas.drop_duplicates()``.
    combined_dbs['temp_order'] = range(combined_dbs.shape[0])

    # Analyze which rows to drop
    if rows_to_conserve_func is not None:
        combined_dbs = combined_dbs[combined_dbs.apply(rows_to_conserve_func, axis=1)]
    
    # Apply post merge mapping
    if isinstance(post_concat_mapping, (list, tuple)) and len(post_concat_mapping) == 3:
        column_name, column_to_extract, func = post_concat_mapping
        combined_dbs[column_name] = func(combined_dbs[column_to_extract].tolist())

    # Sort by ``pull_time_column_name``
    combined_dbs = combined_dbs.sort_values(pull_time_column_name)

    # Convert 'query_column_name' to a tuple of tuples
    # (making them hashable, as required by ``pandas.drop_duplicates()``).
    combined_dbs[query_column_name] = combined_dbs[query_column_name].map(dict_to_tot, na_action='ignore')

    # Remove the 'shared_img_ref' column from consideration
    duplicates_subset_columns_cleaned = [c for c in duplicates_subset_columns if c != 'shared_img_ref']

    # Sort columns with iterables
    if isinstance(columns_with_iterables_to_sort, (list, tuple)):
        for c in columns_with_iterables_to_sort:
            combined_dbs[c] = combined_dbs[c].map(lambda x: tuple(sorted(x)), na_action='ignore')

    # Drop Duplicates (keeping the most recent).
    combined_dbs = combined_dbs.drop_duplicates(subset=duplicates_subset_columns_cleaned, keep='last')

    # Convert the 'query_column_name' dicts back to dictionaries
    combined_dbs[query_column_name] = combined_dbs[query_column_name].map(dict, na_action='ignore')

    # Sort
    combined_dbs = combined_dbs.sort_values('temp_order')
    del combined_dbs['temp_order']

    # Map relationships in the dataframe.
    if relationship_mapping_func is not None:
        combined_dbs = relationship_mapping_func(combined_dbs)

    return combined_dbs.reset_index(drop=True)
























