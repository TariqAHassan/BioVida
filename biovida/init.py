"""

    BioVida Init
    ~~~~~~~~~~~~

    Contains the tools required to construct the caches needed by BioVida.

"""
# Imports
import os


def _sub_directory_creator(root_path, to_create):
    """

    :param root_path:
    :param to_create:
    :return:
    """
    # Init
    trunc_path = None
    created_dirs = dict()

    # Create sub directories
    for sub_dir in to_create:
        if not os.path.isdir(os.path.join(root_path, sub_dir)):
            # Create sub_dir
            os.makedirs(os.path.join(root_path, sub_dir))
            # Record sub_dir's full path
            created_dirs[(sub_dir, True)] = (os.sep).join(os.path.join(root_path, sub_dir).split(os.sep)[-2:])
        else:
            created_dirs[(sub_dir, False)] = (os.sep).join(os.path.join(root_path, sub_dir).split(os.sep)[-2:])

    return created_dirs


def _created_notice(created_list, system_path):
    """

    :param created_list:
    :param system_path:
    :return:
    """
    if len(created_list):
        print("The following directories were created:\n\n%s\nin: '%s'." % \
              ("".join(["  - " + i + "\n" for i in created_list]), system_path + os.sep))
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
    sub_dirs_made = _sub_directory_creator(root_path, ['search_cache', 'image_cache', 'genomic_cache', 'diagnostic_cache'])

    # Record Created Dirs
    created_dirs += {k: v for k, v in sub_dirs_made.items() if k[1] is True}.values()

    # Print results, if verbose is True
    if verbose and len(created_dirs):
        _created_notice(created_dirs, base_path)

    return root_path


def _package_cache_creator(sub_dir, to_create, cache_path=None, verbose=True):
    """

    :param sub_dir: e.g., 'image' (do not include "_cache").
                    Must be one of: 'search_cache', 'image_cache', 'genomic_cache', 'diagnostic_cache'.
    :param cache_path:
    :param to_create:
    :param verbose:
    :return:
    """
    if not isinstance(to_create, (list, tuple)) or not len(to_create):
        raise AttributeError("`to_create` must be an iterable with a nonzero length.")

    # Create main
    root_path = _directory_creator(cache_path, verbose)

    # The full path to the
    sub_dir_full_path = os.path.join(root_path, sub_dir.replace("/", "").strip() + "_cache")

    # Ask for sub directories to be created
    package_created_dirs = _sub_directory_creator(sub_dir_full_path, to_create)

    # New dirs created
    new = {k: v for k, v  in package_created_dirs.items() if k[1] is True}

    # Print record of files created, if verbose is True
    if verbose and len(new.values()):
        _created_notice(new.values(), root_path)

    # Render a hash map of `cache_path` - to - local address
    record_dict = {k[0]: os.path.join(sub_dir_full_path, v.split(os.sep)[-1]) for k, v in package_created_dirs.items()}

    # Return full path & the above mapping
    return sub_dir_full_path, record_dict















