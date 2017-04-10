# coding: utf-8

"""

    Disease-Symptoms Interface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import pandas as pd

# General Support Tools
from biovida.support_tools.support_tools import cln, header
from biovida.support_tools._cache_management import package_cache_creator


class DiseaseSymptomsInterface(object):
    """

    Tools to obtain databases relating symptoms to diseases.

    References:

    - Daniel Himmelstein, Antoine Lizee, Chrissy Hessler, Leo Brueggeman, Sabrina Chen, Dexter Hadley, Ari Green,
      Pouya Khankhanian, Sergio Baranzini. Rephetio: Repurposing drugs on a hetnet [report]. Thinklab (2016).
      doi:`10.15363/thinklab.a7 <http://www.thinklab.com/p/rephetio/report#edges>`_. Online Repository:
      https://github.com/dhimmel/medline.

    - Zhou, XueZhong, Jörg Menche, Albert-László Barabási, and Amitabh Sharma. Human symptoms–disease network.
      Nature communications 5 (2014). doi:`10.1038/ncomms5212 <http://www.nature.com/articles/ncomms5212>`_.
      Online Repository: https://github.com/dhimmel/hsdn.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
                       Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    """

    def __init__(self, cache_path=None, verbose=True):
        self._verbose = verbose
        # Cache creation
        pcc = package_cache_creator(sub_dir='diagnostics',
                                    to_create=['disease_symptoms'],
                                    cache_path=cache_path,
                                    verbose=verbose)
        self.root_path, self._created_disease_ont_dirs = pcc

        # URLs to the databases
        self._hsdn_url = "https://raw.githubusercontent.com/LABrueggs/HSDN/master/Combined-Output.tsv"
        self._rephetio_ml_url = "https://github.com/dhimmel/medline/raw/0c9e2905ccf8aae00af5217255826fe46cba3d30/data/disease-symptom-cooccurrence.tsv"

        self.hsdn_db = None
        self.rephetio_ml_db = None
        self.combined_db = None

    @staticmethod
    def _clean_columns(columns):
        """

        Convert ``columns`` into snake_case.

        :param columns: a list of dataframe columns
        :type columns: ``Pandas Series`` or ``list``
        :return: columns in snake_case
        :rtype: ``list``
        """
        return list(map(lambda x: cln(x).replace(" ", "_").lower(), list(columns)))

    @staticmethod
    def _comma_reverse(term):
        """

        Reverse a single comma, e.g., 'car, blue' --> 'blue car'.

        :param term: a string
        :type term: ``str``
        :return: ``term`` with the comma reversed, if present.
        :rtype: ``str``
        """
        if not isinstance(term, str):
            return term
        elif term.count(", ") != 1:
            return cln(term)
        else:
            return " ".join(cln(term).split(", ")[::-1])

    @staticmethod
    def _harvest(url, cleaner_func, save_path, download_override):
        """

        Pull (i.e., download) a Database.

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
        :type download_override: ``bool``
        :return: the Human Symptoms Disease Network database as a DataFrame.
        :rtype: ``Pandas DataFrame``
        """
        if not os.path.isfile(save_path) or download_override:
            to_return = pd.read_csv(url, sep="\t", header=0, index_col=0).reset_index(drop=True)
            to_return = cleaner_func(to_return)
            to_return.to_pickle(save_path)
        else:
            to_return = pd.read_pickle(save_path)

        return to_return

    def _hsdn_df_cleaner(self, data_frame):
        """

        Clean the harvested HSDN Database in two ways:

        - Lower convert column names to snake_case and lowercase

        - Reverse commas, e.g., 'Arthritis, Rheumatoid' --> 'Rheumatoid Arthritis' (column name: 'common_disease_name').

        :param data_frame: the dataframe obtained in the ``hsdn_pull()`` method.
        :type data_frame: ``Pandas DataFrame``
        """
        # Correct the column names
        data_frame.columns = self._clean_columns(data_frame.columns)

        # Reverse commas
        data_frame['common_disease_name'] = data_frame['mesh_disease_term'].map(self._comma_reverse)
        data_frame['common_symptom_term'] = data_frame['mesh_symptom_term'].map(self._comma_reverse)

        return data_frame

    def hsdn_pull(self, download_override=False):
        """

        Pull (i.e., download) the Human Symptoms Disease Network Database.

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
        :type download_override: ``bool``
        :return: the Human Symptoms Disease Network database as a DataFrame.
        :rtype: ``Pandas DataFrame``
        """
        save_path = os.path.join(self._created_disease_ont_dirs['disease_symptoms'], 'hsdn_db.p')

        if self._verbose:
            header("Downloading Human Symptoms Disease Network Database... ", flank=False)

        # Harvest the database
        self.hsdn_db = self._harvest(url=self._hsdn_url,
                                     cleaner_func=self._hsdn_df_cleaner,
                                     save_path=save_path,
                                     download_override=download_override)

        return self.hsdn_db

    def _rephetio_ml_df_cleaner(self, data_frame):
        """

        Clean the harvested Rephetio Medline Database in two ways:

        - Lower convert column names to snake_case and lowercase

        - Reverse commas, e.g., 'Arthritis, Rheumatoid' --> 'Rheumatoid Arthritis' (column name: 'common_disease_name').

        :param data_frame: the dataframe obtained in the ``rephetio_ml_pull()`` method.
        :type data_frame: ``Pandas DataFrame``
        :return: a cleaned ``data_frame``
        :rtype: ``Pandas DataFrame``
        """
        # Correct the column names
        data_frame.columns = self._clean_columns(data_frame.columns)

        # Reverse commas
        data_frame['common_symptom_term'] = data_frame['mesh_name'].map(self._comma_reverse)

        return data_frame

    def rephetio_ml_pull(self, download_override=False):
        """

        Pull (i.e., download) the Rephetio Medline Database.

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
        :type download_override: ``bool``
        :return: the Rephetio Medline database as a DataFrame.
        :rtype: ``Pandas DataFrame``
        """
        save_path = os.path.join(self._created_disease_ont_dirs['disease_symptoms'], 'rephetio_medline_db.p')

        if self._verbose:
            header("Downloading Rephetio Medline Database... ", flank=False)

        # Harvest the database
        self.rephetio_ml_db = self._harvest(url=self._rephetio_ml_url,
                                            cleaner_func=self._rephetio_ml_df_cleaner,
                                            save_path=save_path,
                                            download_override=download_override)

        return self.rephetio_ml_db

    def _combine(self, download_override, **kwargs):
        """

        Combine `Rephetio Medline` and `Human Symptoms Disease Network` databases.

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
        :type download_override: ``bool``
        :return: a combined dataframe with the following columns: 'common_disease_name' and 'common_symptom_term'.
        :rtype: ``Pandas DataFrame``
        """
        if 'hsdn' in kwargs:
            hsdn = kwargs.get('hsdn')
        else:
            hsdn = self.hsdn_pull(download_override).copy(deep=True)

        if 'rephetio' in kwargs:
            rephetio = kwargs.get('rephetio')
        else:
            rephetio = self.rephetio_ml_pull(download_override).copy(deep=True)

        # Rename `rephetio` columns to s.t. they match with `hsdn`.
        rephetio = rephetio.rename(columns={'doid_name': 'common_disease_name'})

        if self._verbose:
            header("Combining Databases... ", flank=False)

        concat_on_cols = ['common_disease_name', 'common_symptom_term']
        combined_db = pd.concat([hsdn[concat_on_cols], rephetio[concat_on_cols]], ignore_index=True)

        return combined_db.drop_duplicates().reset_index(drop=True)

    def pull(self, download_override=False):
        """

        Construct a dataframe by combining the `Rephetio Medline` and `Human Symptoms Disease Network` databases.

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
        :type download_override: ``bool``
        :return: a combined dataframe with the following columns: 'common_disease_name' and 'common_symptom_term'.
        :rtype: ``Pandas DataFrame``
        """
        save_path = os.path.join(self._created_disease_ont_dirs['disease_symptoms'], 'combined_ds_db.p')

        if not os.path.isfile(save_path) or download_override:
            self.combined_db = self._combine(download_override)
            self.combined_db.to_pickle(save_path)
        else:
            self.combined_db = pd.read_pickle(save_path)

        return self.combined_db
