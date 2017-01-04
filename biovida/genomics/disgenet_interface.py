"""

    DisGeNET Interface
    ~~~~~~~~~~~~~~~~~~


"""
# Imports
import os
import requests
import pandas as pd

# Tool to create required caches
from biovida.init import _package_cache_creator

# BioVida Support Tools
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import camel_to_snake_case

# Cache Creation
cache_path = '/Users/tariq/Google Drive/Programming Projects/BioVida'
root_gene_path, created_gene_dirs = _package_cache_creator(sub_dir='genomics', cache_path=cache_path, to_create=['disgenet'])


def disgenet_disclaimer(print_disclaimer=True):
    """

    Prints or Returns the DisGeNET Disclaimer.

    :param print_disclaimer: if True, print the disclaimer else return.
    :type print_disclaimer: ``bool``
    :return: the DisGeNET Disclaimer.
    :rtype: ``str``
    """
    disclaimer = """

    Disclaimer

    Except where expressly provided otherwise, the site, and all content, materials, information, software, products
    and services provided on the site, are provided on an "as is" and "as available" basis.
    The IBIgroup expressly disclaims all warranties of any kind, whether express or implied, including, but not
    limited to, the implied warranties of merchantability, fitness for a particular purpose and non-infringement.
    The IBI group makes no warranty that:

        a. the site will meet your requirements

        b. the site will be available on an uninterrupted, timely, secure, or error-free basis (though IBI will
           undertake best-efforts to ensure continual uptime and availability of its content)

        c. the results that may be obtained from the use of the site or any services offered through the site will
           be accurate or reliable

        d. the quality of any products, services, information, or other material obtained by you through the site
           will meet your expectations

    Any content, materials, information or software downloaded or otherwise obtained through the use of the site is
    done at your own discretion and risk. The IBI group shall have no responsibility for any damage to your computer
    system or loss of data that results from the download of any content, materials, information or software.
    The IBI group reserves the right to make changes or updates to the site at any time without notice.

    If you have any further questions, please email us at support@disgenet.org
    """
    if print_disclaimer:
        print(disclaimer)
    else:
        return disclaimer


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

    # Write if file does not exist
    if not os.path.isdir(save_address):
        with open(save_address, 'wb') as f:
            f.write(r.content)


def disgenet_database(verbose=False, download_override=False):
    """

    Tool to download a DisGeNET Database.

    ToDo: only 'All Gene Disease Associations' database currently -- include other

    :param verbose: If True, print notice when downloading database.
    :type verbose: ``bool``
    :param download_override: If True, override any existing database currently cached and download a new one.
                              Defaults to False.
    :type download_override: ``bool``
    :return: the DisGeNET 'All Gene Disease Associations' Database
    :rtype: ``Pandas DataFrame``
    """
    # Save address
    save_address = os.path.join(created_gene_dirs['disgenet'], 'all_associations.csv')

    # Download Location
    db_url = 'http://www.disgenet.org/ds/DisGeNET/results/all_gene_disease_associations.tsv.gz'

    # Download or simply load from cache
    if not os.path.isfile(save_address) and not download_override:
        if verbose:
            header("Downloading DisGeNET Database... ", flank=False)
        # Harvest, Rename Columns and Save
        df = pd.read_csv(db_url, sep='\t', header=21, compression='gzip')
        df.columns = list(map(camel_to_snake_case, df.columns))
        df.to_csv(save_address, index=False)
    else:
        df = pd.read_csv(save_address)

    return df




























































