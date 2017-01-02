"""

    BioVida Init
    ~~~~~~~~~~~~


"""
# Imports
import os


def _directory_creator(cache_path=None, verbose=True):
    """

    Tool to create directories needed by BioVida.

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
        cache_path_clean = (cache_path if not cache_path.endswith("/") else cache_path[:-1])
    else:
        cache_path_clean = cache_path
    
    # Set the base path to the home directory if `cache_path_clean` does not exist
    if not (isinstance(cache_path_clean, str) and os.path.isdir(cache_path_clean)):
        base_path = os.path.expanduser("~")
    elif isinstance(cache_path_clean, str) and not os.path.isdir(cache_path_clean):
        raise FileNotFoundError("[Errno 2] No such file or directory: '%s'." % (cache_path_clean))
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
    trunc_path = None
    for sub_dir in ['image_cache', 'genomic_cache', 'diagnostic_cache']:
        if not os.path.isdir(os.path.join(root_path, sub_dir)):
            # Create sub_dir
            os.makedirs(os.path.join(root_path, sub_dir))
            # Trunc sub_dir
            trunc_path = (os.sep).join(os.path.join(root_path, sub_dir).split(os.sep)[-2:])
            # Note its creation
            created_dirs.append(trunc_path)

    # Print results, if verbose is True
    if verbose and len(created_dirs):
        print("The following directories were created:\n\n%s\nin: '%s'." % \
              ("".join(["  - " + i + "\n" for i in created_dirs]), base_path + "/"))

    return root_path





