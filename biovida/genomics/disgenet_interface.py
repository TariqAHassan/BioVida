"""

    DisGeNET Interface
    ~~~~~~~~~~~~~~~~~~


"""
# Imports
import os
import requests
import pandas as pd
from pprint import pprint

# Tool to create required caches
from biovida.support_tools._cache_management import _package_cache_creator

# BioVida Support Tools
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import camel_to_snake_case
from biovida.support_tools.support_tools import list_to_bulletpoints

# BioVida Printing Tools
from biovida.support_tools.printing import dict_pretty_printer


# ---------------------------------------------------------------------------------------------
# DisGeNET Reference Data
# ---------------------------------------------------------------------------------------------


_disgenet_delimited_databases = {
    # Source: http://www.disgenet.org/web/DisGeNET/menu/downloads#curated
    # Structure: {database_short_name: {full_name: ..., url: ..., description: ..., number_of_rows_in_header: ...}}
    'all': {
        'full_name': 'All Gene-Disease Associations',
        'url': 'http://www.disgenet.org/ds/DisGeNET/results/all_gene_disease_associations.tsv.gz',
        'description': 'The file contains all gene-disease associations in DisGeNET.',
        'header': 21
    },
    'curated': {
        'full_name': 'Curated Gene-Disease Associations',
        'url': 'http://www.disgenet.org/ds/DisGeNET/results/curated_gene_disease_associations.tsv.gz',
        'description': 'The file contains gene-disease associations from UNIPROT, CTD (human subset), ClinVar, Orphanet,'
                       ' and the GWAS Catalog.',
        'header': 21
    },
    'snp_disgenet': {
        'full_name': 'All SNP-Gene-Disease Associations',
        'url': 'http://www.disgenet.org/ds/DisGeNET/results/all_snps_sentences_pubmeds.tsv.gz',
        'description': 'All SNP-gene-disease associations.',
        'header': 20
    },
}


# ---------------------------------------------------------------------------------------------
# Tools for Harvesting DisGeNET Data
# ---------------------------------------------------------------------------------------------


def _disgenet_readme(created_gene_dirs):
    """

    Writes the DisGeNET README to disk.

    :param created_gene_dirs: the dictionary of directories returned by ``_package_cache_creator()``
    :type created_gene_dirs: ``dict``
    """
    # Save address
    save_address = os.path.join(created_gene_dirs['disgenet'], 'DisGeNET_README.txt')

    # Write if the file does not exist
    if not os.path.isfile(save_address):
        # Resquest
        r = requests.get(readme_url, stream=True)

        # URL to DisGeNET README
        readme_url = 'http://www.disgenet.org/ds/DisGeNET/results/readme.txt'

        # Write
        with open(save_address, 'wb') as f:
            f.write(r.content)

        # Notice
        header("The DisGeNET README has been downloaded to:\n\n {0}\n\n"
               "Please take the time to review this document.".format(save_address))


class DisgenetInterface(object):
    """

    Python Interface for Harvesting Databases from DisGeNET.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
                       Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    """

    def __init__(self, cache_path=None, verbose=True):
        """

        Initialize the ``DisgenetInterface()`` Class.

        """
        self._verbose = verbose

        # Cache Creation
        ppc = _package_cache_creator(sub_dir='genomics', cache_path=cache_path, to_create=['disgenet'])
        self.root_path, self._created_gene_dirs = ppc

        # Check if a readme exists.
        _disgenet_readme(self._created_gene_dirs)

        # Containers for the most recently requested database.
        self.current_database = None
        self.current_database_name = None
        self.current_database_full_name = None
        self.current_database_description = None

    def _disgenet_delimited_databases_key_error(self, database):
        """

        Raises an error when an reference is made to a database not in `_disgenet_delimited_databases.keys()`.

        :param database: `erroneous` database reference.
        :type database: ``str``
        """
        if database not in _disgenet_delimited_databases:
            raise ValueError("'{0}' is an invalid value for `database`.\n`database` must be one of:\n{1}".format(
                str(database), list_to_bulletpoints(_disgenet_delimited_databases.keys())))

    def options(self, database=None, pretty_print=True):
        """

        Disgenet databases which can be downloaded
        as well as additional information about the databases.

        :param database: A database to review. Must be one of: 'all', 'curated', 'snp_disgenet' or ``None``.
                         If a specific database is given, the database's full name and description will be provided.
                         If ``None``, a list of databases which can be downloaded will be returned (or printed).
                         Defaults to ``None``.
        :type database: ``str``
        :param pretty_print: pretty print the information. Defaults to True.
        :type pretty_print: ``bool``
        :return: a ``list`` if `database` is ``None``, else a ``dict`` with the database's full name and description.
        :rtype: ``list`` or ``dict``
        """
        if database is None:
            info = list(_disgenet_delimited_databases.keys())
        elif database in _disgenet_delimited_databases:
            info = {k: v for k, v in _disgenet_delimited_databases[database].items() if k in ['full_name', 'description']}
        else:
            self._disgenet_delimited_databases_key_error(database)

        if pretty_print:
            if database is None:
                print("Available Databases:\n")
                print(list_to_bulletpoints(info))
            else:
                dict_pretty_printer(info)
        else:
            return info

    def pull(self, database, download_override=False, snake_case_col_names=False):
        """

        Pull (i.e., download) a DisGeNET Database.

        Note: if a database is already cached, it will be used instead of downloading
        (the `download_override` argument can be used override this behaviour).

        :param database: A database to download. Must be one of: 'all', 'curated', 'snp_disgenet' or ``None``.
                         See ``options()`` for more information.
        :type database: ``str``
        :param download_override: If True, override any existing database currently cached and download a new one.
                                  Defaults to False.
        :type download_override: ``bool``
        :param snake_case_col_names: if True, convert column names to 'snake case' (e.g., 'this_is_snake_case').
                                     Defaults to False (which will leave the column names in `camelCase`).
        :type snake_case_col_names: ``bool``
        :return: a DisGeNET database
        :rtype: ``Pandas DataFrame``
        """
        self._disgenet_delimited_databases_key_error(database)

        # Download Location
        db_url = _disgenet_delimited_databases[database]['url']

        # Save Name
        save_name = "{0}.csv".format(db_url.split("/")[-1].split(".")[0])

        # Save address
        save_address = os.path.join(self._created_gene_dirs['disgenet'], save_name)

        # Download or simply load from cache
        if download_override or not os.path.isfile(save_address):
            if self._verbose:
                header("Downloading DisGeNET Database... ", flank=False)
            # Harvest and Save
            df = pd.read_csv(db_url, sep='\t', header=_disgenet_delimited_databases[database]['header'], compression='gzip')
            df.to_csv(save_address, index=False)
        else:
            df = pd.read_csv(save_address)

        #  Rename Columns, if requested
        if snake_case_col_names:
            df.columns = list(map(camel_to_snake_case, df.columns))

        # Cache the database
        self.current_database = df
        self.current_database_name = database
        self.current_database_full_name = _disgenet_delimited_databases[database]['full_name']
        self.current_database_description = _disgenet_delimited_databases[database]['description']

        return df



































