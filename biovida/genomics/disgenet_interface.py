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
from biovida.init import _package_cache_creator

# BioVida Support Tools
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import camel_to_snake_case
from biovida.support_tools.support_tools import list_to_bulletpoints

# BioVida Printing Tools
from biovida.support_tools.printing import dict_pretty_printer


# Cache Creation
cache_path = '/Users/tariq/Google Drive/Programming Projects/BioVida'
root_gene_path, created_gene_dirs = _package_cache_creator(sub_dir='genomic', cache_path=cache_path, to_create=['disgenet'])


_disgenet_delimited_databases = {
    # Source: http://www.disgenet.org/web/DisGeNET/menu/downloads#curated
    # Structure: {database_short_name: {full_name: ..., url: ..., description: ..., number_of_rows_in_header: ...}}
    'curated': {
        'full_name': 'Curated Gene-Disease Associations',
        'url': 'http://www.disgenet.org/ds/DisGeNET/results/curated_gene_disease_associations.tsv.gz',
        'description': 'The file contains gene-disease associations from UNIPROT, CTD (human subset), ClinVar, Orphanet,'
                       ' and the GWAS Catalog.',
        'header': 21
    },
    'all': {
        'full_name': 'All Gene-Disease Associations',
        'url': 'http://www.disgenet.org/ds/DisGeNET/results/all_gene_disease_associations.tsv.gz',
        'description': 'The file contains all gene-disease associations in DisGeNET.',
        'header': 21
    },
    'snp_disgenet': {
        'full_name': 'All SNP-Gene-Disease Associations',
        'url': 'http://www.disgenet.org/ds/DisGeNET/results/all_snps_sentences_pubmeds.tsv.gz',
        'description': 'All SNP-gene-disease associations.',
        'header': 20
    },
}


def _disgenet_readme():
    """
    Writes the DisGeNET README to disk.
    """
    # URL to readme
    readme_url = 'http://www.disgenet.org/ds/DisGeNET/results/readme.txt'

    # Resquest
    r = requests.get(readme_url, stream=True)

    # Save address
    save_address = os.path.join(created_gene_dirs['disgenet'], 'DisGeNET_README.txt')

    # Write if the file does not exist
    if not os.path.isdir(save_address):
        with open(save_address, 'wb') as f:
            f.write(r.content)


def _disgenet_delimited_databases_key_error(database):
    """

    :param database:
    :return:
    """
    if database not in _disgenet_delimited_databases:
        raise ValueError("'{0}' is an invalid value for `database`.\n`database` must be one of:\n{1}".format(
            str(database), list_to_bulletpoints(_disgenet_delimited_databases.keys())))


def disgenet_options(database, pretty_print=False):
    """

    Options for disgenet Databases.
    Provides the full name

    :param database: one of 'curated', 'all' or 'snp_disgenet'.
                     if ``None``, return (or print) a list of databases which can be downloaded.
                     if a specific database is given, the database's full name and description will be provided.
    :type database: ``str``
    :param pretty_print: pretty print the information.
    :type pretty_print: ``bool``
    :return: a ``list`` if `database` is ``None``, else ``dict`` with the database's full name and description.
    :rtype: ``list`` or ``dict``
    """
    if database is None:
        info = list(_disgenet_delimited_databases.keys())
    elif database in _disgenet_delimited_databases:
        info = {k: v for k, v in _disgenet_delimited_databases[database].items() if k in ['full_name', 'description']}
    else:
        _disgenet_delimited_databases_key_error(database)

    if pretty_print:
        if database is None:
            print("Available Databases:\n")
            print(list_to_bulletpoints(info))
        else:
            dict_pretty_printer(info)
    else:
        return info


def disgenet_database(database='all', verbose=False, download_override=False, snake_case_col_names=False):
    """

    Tool to download a DisGeNET Database.

    :param database: A database to download. Must be one of: 'curated', 'all' or 'snp_disgenet'.
                     See ``disgenet_options()`` for more information. Defaults to 'all'.
    :type database: ``str``
    :param verbose: If True, print notice when downloading database.
    :type verbose: ``bool``
    :param download_override: If True, override any existing database currently cached and download a new one.
                              Defaults to False.
    :type download_override: ``bool``
    :param snake_case_col_names: if True, convert column names to 'snake case' (e.g., 'this_is_snake_case').
                                 Defaults to False (which will leave the column names in `camelCase`).
    :type snake_case_col_names: ``bool``
    :return: a DisGeNET database
    :rtype: ``Pandas DataFrame``
    """
    _disgenet_delimited_databases_key_error(database)

    # Download Location
    db_url = _disgenet_delimited_databases[database]['url']

    # Save Name
    save_name = "{0}.csv".format(db_url.split("/")[-1].split(".")[0])

    # Save address
    save_address = os.path.join(created_gene_dirs['disgenet'], save_name)

    # Download or simply load from cache
    if download_override or not os.path.isfile(save_address):
        if verbose:
            header("Downloading DisGeNET Database... ", flank=False)
        # Harvest and Save
        df = pd.read_csv(db_url, sep='\t', header=_disgenet_delimited_databases[database]['header'], compression='gzip')
        df.to_csv(save_address, index=False)
    else:
        df = pd.read_csv(save_address)

    #  Rename Columns, if requested
    if snake_case_col_names:
        df.columns = list(map(camel_to_snake_case, df.columns))

    return df













































