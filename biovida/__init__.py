__version_info__ = (0, 1, 1)
__version__ = '.'.join(map(str, __version_info__))
__version_numeric__ = sum([i / 10**e for e, i in enumerate(__version_info__)])
