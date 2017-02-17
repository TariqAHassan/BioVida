"""

    General Tools for Image Processing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
# from keras.preprocessing import image
from scipy.misc import imread, imresize
from skimage.color.colorconv import rgb2gray


class NoResultsFound(Exception):
    pass


def dict_to_tot(d):
    """

    Convert a dictionary to a tuple of tuples and sort by the former keys.

    :param d:
    :return:
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

    :param converted_image:
    :param img_size:
    :param axes:
    :return:
    """
    return np.transpose(imresize(converted_image, img_size), axes).astype('float32')


def load_and_scale_imgs(list_of_images, img_size, axes=(2, 0, 1), status=True, grayscale_first=False):
    """

    :param list_of_images:
    :param img_size:
    :param axes:
    :param status:
    :param grayscale_first: convert the image to grayscale first.
    :return:
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


def load_temp_dbs(temp_db_path):
    """

    Load temporary databases in the 'databases/__temp__' directory.

    :param temp_db_path: path to the temporary databases.
    :type temp_db_path: ``str``
    :return:
    """
    # Apply the merging function
    db_paths = [os.path.join(temp_db_path, p) for p in os.listdir(temp_db_path) if p.endswith(".p")]
    
    if not len(db_paths):
        return None

    # Read the dataframes in the '__temp__' directory into memory
    frames = [pd.read_pickle(p) for p in db_paths]

    # Concatenate all frames
    return pd.concat(frames, ignore_index=True)


def record_db_merge(current_record_db,
                   record_db_addition,
                   query_column_name,
                   query_time_column_name,
                   duplicates_subset_columns,
                   sort_on,
                   relationship_mapping_func=None):
    """

    Merge the existing record database with new additions.

    :param current_record_db: 
    :type current_record_db: ``Pandas DataFrame``
    :param record_db_addition: the new search dataframe to added to the existing one.
    :type record_db_addition: ``Pandas DataFrame``
    :param query_column_name: the column which contains the query responcible for the results. Note: this column
                              *should* contain only dictionaries.
    :type query_column_name: ``str``
    :param query_time_column_name: 
    :type query_time_column_name: ``str``
    :param duplicates_subset_columns: a list (or tuple) of columns to consider when dropping duplicates.
    :type duplicates_subset_columns: ``list`` or ``tuple``
    :param sort_on: 
    :type sort_on: ``str`` or ``None``
    :param relationship_mapping_func: function to map relationships in the dataframe. Defaults to ``None``.
    :type relationship_mapping_func:
    :return:
    :rtype: ``Pandas DataFrame``
    """
    # Load in the current database and combine with the `record_db_addition` database
    combined_dbs = pd.concat([current_record_db, record_db_addition], ignore_index=True)

    # Sort by 'QueryDate'
    combined_dbs = combined_dbs.sort_values(query_time_column_name)

    # Convert 'query_column_name' to a tuple of tuples
    # (making them hashable, as required by ``pandas.drop_duplicates()``).
    combined_dbs[query_column_name] = combined_dbs[query_column_name].map(dict_to_tot, na_action='ignore')

    # Remove the 'shared_img_ref' column from consideration
    duplicates_subset_columns_cleaned = [c for c in duplicates_subset_columns if c != 'shared_img_ref']

    # Drop Duplicates (keeping the most recent).
    combined_dbs = combined_dbs.drop_duplicates(subset=duplicates_subset_columns_cleaned, keep='last')

    # Convert the 'query_column_name' dicts back to dictionaries
    combined_dbs[query_column_name] = combined_dbs[query_column_name].map(dict, na_action='ignore')

    # Save to class instance
    if sort_on is not None:
        current_record_db = combined_dbs.sort_values(sort_on)
    else:
        current_record_db = combined_dbs

    # Map relationships in the dataframe.
    if relationship_mapping_func is not None:
        current_record_db = relationship_mapping_func(current_record_db)

    return current_record_db.reset_index(drop=True)
























