"""

    BioVida Init
    ~~~~~~~~~~~~


"""
# Imports
import os


def _sub_directory_creator(root_path, to_create):
    """

    :param root_path:
    :param to_create:
    :return:
    """
    created_dirs = list()
    trunc_path = None

    # Create sub directories
    for sub_dir in to_create:
        if not os.path.isdir(os.path.join(root_path, sub_dir)):
            # Create sub_dir
            os.makedirs(os.path.join(root_path, sub_dir))
            # Trunc sub_dir
            trunc_path = (os.sep).join(os.path.join(root_path, sub_dir).split(os.sep)[-2:])
            # Note its creation
            created_dirs.append(trunc_path)

    return created_dirs


def _created_notice(created_list, system_path):
    """

    :param created_list:
    :param system_path:
    :return:
    """
    if len(created_list):
        print("The following directories were created:\n\n%s\nin: '%s'." % \
              ("".join(["  - " + i + "\n" for i in created_list]), system_path + "/"))
        print("\n")


def _directory_creator(cache_path=None, verbose=True):
    """

    Tool to create directories needed by BioVida.

    Required:
      - biovida_cache
      - biovida_cache/search_cache
      - biovida_cache/image_cache
      - biovida_cache/genomic_cache
      - biovida_cache/diagnostic_cache

    :param cache_path: path to create to create the `BioVida` cache.
                       If ``None``, the home directory will be used. Defaults to None.
    :type cache_path: ``str`` or ``None``
    :param verbose: Notify user of directory that have been created. Defaults to True.
    :type verbose: ``bool``
    :return:
    """
    # Init. Root Path
    root_path = None

    # Record of dirs created.
    created_dirs = list()

    # Clean `cache_path`
    if cache_path is not None and cache_path != None:
        cache_path_clean = (cache_path.strip() if not cache_path.endswith("/") else cache_path.strip()[:-1])
    else:
        cache_path_clean = cache_path

    # Set the base path to the home directory if `cache_path_clean` does not exist
    if isinstance(cache_path_clean, str) and not os.path.isdir(cache_path_clean):
        raise FileNotFoundError("[Errno 2] No such file or directory: '%s'." % (cache_path_clean))
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

    # Check if 'image', 'genomic' and 'diagnostic' caches exist, if not create them.
    created_dirs += _sub_directory_creator(root_path, ['search_cache', 'image_cache', 'genomic_cache', 'diagnostic_cache'])

    # Print results, if verbose is True
    if verbose and len(created_dirs):
        _created_notice(created_dirs, base_path)

    return root_path


def _package_cache_creator(sub_dir, cache_path=None, to_create=None, verbose=True):
    """

    :param sub_dir: e.g., 'image' (do not include "_cache").
    :param cache_path:
    :param to_create:
    :param verbose:
    :return:
    """
    # Create main
    root_path = _directory_creator(cache_path, verbose)

    # Create
    if to_create is not None:
        package_created_dirs = _sub_directory_creator(os.path.join(root_path, sub_dir + "_cache"), to_create)
    else:
        package_created_dirs = []

    # Print record of files created
    _created_notice(package_created_dirs, root_path)















