# coding: utf-8

"""

    BioVida Cache Management Tools
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Note: this module contains the tools required to construct the caches needed by BioVida.
import os
import requests
from PIL import Image

# General Support Tools
from biovida.support_tools.support_tools import combine_dicts, list_to_bulletpoints


def _medpix_logo_download(save_path,
                          verbose,
                          image_web_address='https://openi.nlm.nih.gov/imgs/512/341/1544/MPX1544_synpic54565.png',):
    """

    Download and save the MedPix logo from a representative image.
    (The image is needed for image processing).

    :param image_web_address: the location of the image which contains the MedPix Logo
    :type image_web_address: ``str``
    :param verbose: notify user of directory that have been created. Defaults to True.
    :type verbose: ``bool``
    :return: a dictionary of the form {'medpix_logo': PATH TO THE LOGO}.
    :rtype: ``dict``
    """
    full_save_path = os.path.join(save_path, "medpix_logo.png")

    if not os.path.isfile(full_save_path):
        # Get representative images
        image = Image.open(requests.get(image_web_address, stream=True).raw)
        # Crop and Save
        image_cropped = image.crop((406, 6, 502, 27))
        image_cropped.save(full_save_path)
        if verbose:
            print("\nThe MedPix Logo, required for processing images from Open-i, "
                  "has been downloaded to:\n\n{0}\n".format(list_to_bulletpoints([full_save_path])))

    return {'medpix_logo': full_save_path}


def _sub_directory_creator(root_path, to_create):
    """

    :param root_path:
    :param to_create:
    :return:
    """
    created_dirs = dict()

    # Create sub directories
    for sub_dir in to_create:
        # Check if the directory exists
        if not os.path.isdir(os.path.join(root_path, sub_dir)):
            os.makedirs(os.path.join(root_path, sub_dir))
            # Record sub_dir's full path
            created_dirs[(sub_dir, True)] = (os.sep).join(os.path.join(root_path, sub_dir).split(os.sep)[-2:])
        else:
            # Note that it was not created
            created_dirs[(sub_dir, False)] = (os.sep).join(os.path.join(root_path, sub_dir).split(os.sep)[-2:])

    return created_dirs


def _created_notice(created_list, system_path):
    """

    :param created_list:
    :param system_path:
    :return:
    """
    if len(created_list):
        print("The following directories were created:\n%s\nin: '%s'." % \
              ("".join(["  - " + i + "\n" for i in created_list]), system_path + os.sep))
        print("\n")


def _directory_creator(cache_path=None, verbose=True):
    """

    Tool to create directories needed by BioVida.

    Required:
      - biovida_cache
      - biovida_cache/images_cache
      - biovida_cache/genomics_cache
      - biovida_cache/diagnostics_cache

    :param cache_path: path to create to create the `BioVida` cache.
                       If ``None``, the home directory will be used. Defaults to None.
    :type cache_path: ``str`` or ``None``
    :param verbose: notify user of directory that have been created. Defaults to True.
    :type verbose: ``bool``
    :return: the root path (i.e., path the cache itself).
    :rtype: ``str``
    """
    # Record of dirs created.
    created_dirs = list()

    # Clean `cache_path`
    if cache_path is not None and cache_path is not None:
        cache_path_clean = (cache_path.strip() if not cache_path.endswith(os.sep) else cache_path.strip()[:-1])
    else:
        cache_path_clean = cache_path

    # Set the base path to the home directory if `cache_path_clean` does not exist
    if isinstance(cache_path_clean, str) and not os.path.isdir(cache_path_clean):
        raise FileNotFoundError("[Errno 2] No such file or directory: '{0}'.".format(cache_path_clean))
    elif not (isinstance(cache_path_clean, str) and os.path.isdir(cache_path_clean)):
        base_path = os.path.expanduser("~")
    else:
        base_path = cache_path_clean

    # Set the 'biovida_cache' directory to be located in the home folder
    root_path = os.path.join(base_path, "biovida_cache")
    # If needed, create
    if not os.path.isdir(root_path):
        # Create main cache folder
        os.makedirs(os.path.join(base_path, "biovida_cache"))
        # Note its creation
        created_dirs.append("biovida_cache")

    # Check if 'search', 'images', 'genomics' and 'diagnostics' caches exist, if not create them.
    sub_dirs_made = _sub_directory_creator(root_path, ['images_cache', 'genomics_cache', 'diagnostics_cache'])

    # Record Created Dirs
    created_dirs += {k: v for k, v in sub_dirs_made.items() if k[1] is True}.values()

    # Print results, if verbose is True
    if verbose and len(created_dirs):
        _created_notice(created_dirs, base_path)

    return root_path


def _add_to_create_nest(nest, record_dict, verbose):
    """

    Add `nest` directories, i.e., create directories within directories delineated in `to_create` (below)..

    :param nest:
    :param record_dict:
    :return:
    :rtype: ``dict``
    """
    created = list()
    if isinstance(nest, (list, tuple)):
        for (to_create_dir, new_nested_dir) in nest:
            new_dir_name = os.path.join(record_dict[to_create_dir], new_nested_dir)
            if not os.path.exists(new_dir_name):
                os.makedirs(new_dir_name)
                created.append(new_dir_name)
            record_dict[new_nested_dir] = new_dir_name

    if verbose and len(created):
        print("The following nested directories were also created:\n{0}\n".format(list_to_bulletpoints(created)))

    return record_dict


def package_cache_creator(sub_dir, to_create, cache_path=None, nest=None, verbose=True, requires_medpix_logo=False):
    """

    Create a cache for a given ``sub_dir``. If no biovida path exists in ``cache_path``,
    all of the following caches will be created: 'search', 'images', 'genomics' and 'diagnostics'.

    :param sub_dir: e.g., 'images' (do not include "_cache").
                    Must be one of: 'search', 'images', 'genomics', 'diagnostics'.
    :type sub_dir: ``str``
    :param to_create: subdirectories within ``sub_dir`` to be created.
    :type to_create: ``iterable``
    :param cache_path: local path to the desired location of the cache.
                       If ``None``, will default to the the home directory. Defaults to ``None``.
    :type cache_path: ``str`` or ``None``
    :param nest: a list of the form [(to_create item, new nested directory name), ...]
    :type nest: ``list of iterables`` or ``None``
    :param verbose: If True, print updates. Defaults to True.
    :type verbose: ``bool``
    :param requires_medpix_logo: if ``True``, download the medpix logo.
    :type requires_medpix_logo: ``bool``
    :return: tuple of the form ``(local path to `sub_dir`, `record_dict`)``,
             where ``record_dict`` is of the form ``{to_create[0]: 'PATH_1', to_create[1]: 'PATH_2', ...}``.
             Note: ``record_dict`` if ``requires_medpix_logo`` is ``True``, a 'medpix_logo' key will also be present.
    :rtype: ``tuple``
    """
    # Check `sub_dir` is an allowed type
    allowed_sub_dirs = ('search', 'images', 'genomics', 'diagnostics')
    if sub_dir not in allowed_sub_dirs:
        raise ValueError("`sub_dir` must be one of:\n{0}".format(list_to_bulletpoints(allowed_sub_dirs)))

    # Check `to_create` is a ``list`` or ``tuple`` with nonzero length
    if not isinstance(to_create, (list, tuple)) or not len(to_create):
        raise AttributeError("`to_create` must be a `list` or `tuple` with a nonzero length.")

    # Create main
    root_path = _directory_creator(cache_path, verbose)

    # The full path to the subdirectory
    sub_dir_full_path = os.path.join(root_path, sub_dir.replace("/", "").strip() + "_cache")

    # Ask for sub directories to be created
    package_created_dirs = _sub_directory_creator(sub_dir_full_path, to_create)

    # New dirs created
    new = {k: v for k, v in package_created_dirs.items() if k[1] is True}

    # Print record of files created, if verbose is True
    if verbose and len(new.values()):
        _created_notice(new.values(), root_path)

    # Render a hash map of `cache_path` - to - local address
    record_dict = {k[0]: os.path.join(sub_dir_full_path, v.split(os.sep)[-1]) for k, v in package_created_dirs.items()}
    record_dict['ROOT_PATH'] = root_path

    # Add nested directories, if any
    record_dict_nest = _add_to_create_nest(nest, record_dict, verbose)
    
    # Download the medpix logo to `sub_dir`, if `sub_dir` is 'images'.
    if sub_dir == 'images' and requires_medpix_logo:
        if 'aux' not in record_dict_nest.keys():
            raise ValueError("`nest` (or `to_create`) must contain 'aux' if `requires_medpix_logo` is `True`.")
        medpix_logo_location = _medpix_logo_download(record_dict_nest['aux'], verbose=verbose)
        # Add the path to the logo to `record_dict_nest`
        record_dict_nest = combine_dicts(record_dict_nest, medpix_logo_location)

    # Return full path & the above mapping
    return sub_dir_full_path, record_dict_nest
