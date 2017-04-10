# coding: utf-8

"""

    DisGeNET Interface
    ~~~~~~~~~~~~~~~~~~

"""
import os
import requests
import pandas as pd

# Tool to create required caches
from biovida.support_tools._cache_management import package_cache_creator

# BioVida Support Tools
from biovida.support_tools.support_tools import header, camel_to_snake_case, list_to_bulletpoints

# BioVida Printing Tools
from biovida.support_tools.printing import dict_pprint


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


class DisgenetInterface(object):
    """

    Python Interface for Harvesting Databases from `DisGeNET <http://www.disgenet.org/>`_.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
                       Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    """

    @staticmethod
    def _disgenet_readme(created_gene_dirs):
        """

        Writes the DisGeNET README to disk.

        :param created_gene_dirs: the dictionary of directories returned by ``_package_cache_creator()``
        :type created_gene_dirs: ``dict``
        """
        save_address = os.path.join(created_gene_dirs['disgenet'], 'DisGeNET_README.txt')

        if not os.path.isfile(save_address):
            readme_url = 'http://www.disgenet.org/ds/DisGeNET/results/readme.txt'
            r = requests.get(readme_url, stream=True)
            with open(save_address, 'wb') as f:
                f.write(r.content)
            header("The DisGeNET README has been downloaded to:\n\n {0}\n\n"
                   "Please take the time to review this document.".format(save_address), flank=False)

    def __init__(self, cache_path=None, verbose=True):
        """

        Initialize the ``DisgenetInterface()`` Class.

        """
        self._verbose = verbose

        # Cache Creation
        ppc = package_cache_creator(sub_dir='genomics',
                                    cache_path=cache_path,
                                    to_create=['disgenet'],
                                    verbose=verbose)
        self.root_path, self._created_gene_dirs = ppc

        # Check if a readme exists.
        self._disgenet_readme(self._created_gene_dirs)

        # Containers for the most recently requested database.
        self.current_database = None
        self.current_database_name = None
        self.current_database_full_name = None
        self.current_database_description = None

    @staticmethod
    def _disgenet_delimited_databases_key_error(database):
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
            info = {k: v for k, v in _disgenet_delimited_databases[database].items()
                    if k in ['full_name', 'description']}
        else:
            self._disgenet_delimited_databases_key_error(database)

        if pretty_print:
            if database is None:
                print("Available Databases:\n")
                print(list_to_bulletpoints(info))
            else:
                dict_pprint(info)
        else:
            return info

    @staticmethod
    def _df_clean(data_frame):
        """
        
        Clean the dataframe generated by ``pull()``

        :param data_frame:
        :type data_frame: ``Pandas DataFrame``
        :return: see description.
        :rtype: ``Pandas DataFrame``
        """
        # Lower to make easier to match in the future
        data_frame['diseaseName'] = data_frame['diseaseName'].map(
            lambda x: x.lower() if isinstance(x, str) else x, na_action='ignore')

        data_frame.columns = list(map(camel_to_snake_case, data_frame.columns))

        return data_frame

    def pull(self, database, download_override=False):
        """

        Pull (i.e., download) a DisGeNET Database.

        Note: if a database is already cached, it will be used instead of downloading
        (the `download_override` argument can be used override this behaviour).

        :param database: A database to download. Must be one of: 'all', 'curated', 'snp_disgenet' or ``None``.
                         See ``options()`` for more information.
        :type database: ``str``
        :param download_override: If ``True``, override any existing database currently cached and download a new one.
                                  Defaults to ``False``.
        :type download_override: ``bool``
        :return: a DisGeNET database
        :rtype: ``Pandas DataFrame``
        """
        self._disgenet_delimited_databases_key_error(database)
        db_url = _disgenet_delimited_databases[database]['url']
        save_name = "{0}.p".format(db_url.split("/")[-1].split(".")[0])
        save_address = os.path.join(self._created_gene_dirs['disgenet'], save_name)

        if download_override or not os.path.isfile(save_address):
            if self._verbose:
                header("Downloading DisGeNET Database... ", flank=False)
            data_frame = pd.read_csv(db_url,
                                     sep='\t',
                                     header=_disgenet_delimited_databases[database]['header'],
                                     compression='gzip')
            self._df_clean(data_frame).to_pickle(save_address)
        else:
            data_frame = pd.read_pickle(save_address)

        # Cache the database
        self.current_database = data_frame
        self.current_database_name = database
        self.current_database_full_name = _disgenet_delimited_databases[database]['full_name']
        self.current_database_description = _disgenet_delimited_databases[database]['description']

        return data_frame
