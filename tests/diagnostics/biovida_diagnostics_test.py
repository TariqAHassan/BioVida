"""

    BioVida-Diagnostics Subpackage Unit Testing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import sys
import pickle
import unittest
import pandas as pd
from os.path import join as os_join

# Allow access to modules
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from biovida.diagnostics import DiseaseOntInterface
from biovida.diagnostics import DiseaseSymptomsInterface


doi = DiseaseOntInterface(verbose=False)
dsi = DiseaseSymptomsInterface(verbose=False)

data_path = os_join(str(os.getcwd()).split("/tests")[0], "tests/diagnostics/data")


class DiseaseOntInterfaceTests(unittest.TestCase):
    """
    
    Unit Tests for the DiseaseOntInterface Class.
    
    """
    def __init__(self, *args, **kwargs):
        # see: http://stackoverflow.com/a/17353262/4898004.
        super(DiseaseOntInterfaceTests, self).__init__(*args, **kwargs)

        with open(os_join(data_path, "obo_file.p"), "rb") as f:
            self.obo_file = pickle.load(f)

    def test_parse(self):
        """Test the the data harvesting/processing engine generates a DataFrame."""
        doi._harvest_engine(disease_ontology_db_url='', obo_file=self.obo_file)
        self.assertEqual(isinstance(doi.disease_db, pd.DataFrame), True)

    def test_correct_dims_db(self):
        """Test that the dimensions of ``doi.disease_db`` are correct."""
        doi._harvest_engine(disease_ontology_db_url='', obo_file=self.obo_file)

        # Columns
        required_cols = ['alt_id', 'comment', 'created_by', 'creation_date', 'def',
                         'def_urls', 'disjoint_from', 'id', 'is_a', 'is_a_doid',
                         'is_obsolete', 'name', 'namespace', 'subset', 'synonym',
                         'synonym_exact', 'synonym_narrow', 'synonym_related', 'xref']
        n_missing_columns = len(set(required_cols) - set(doi.disease_db.columns))
        self.assertEqual(n_missing_columns == 0, True)

        # Rows
        n_terms = self.obo_file.count("[Term]")
        self.assertEqual(doi.disease_db.shape[0] == n_terms, True)


class DiseaseSymptomsInterfaceTests(unittest.TestCase):
    """

    Unit Tests for the DiseaseSymptomsInterface Class.

    """

    @staticmethod
    def _pickle_to_tsv_to_df(file_name, cleaner_func):
        
        with open(os_join(data_path, "{0}.p".format(file_name)), "rb") as f:
            hsdn_file = pickle.load(f)
        
        tsv_file = os_join(data_path, "{0}.tsv".format(file_name))
        
        with open(tsv_file, "w") as f:
            f.write(hsdn_file)
        
        df = dsi._harvest(url=tsv_file, cleaner_func=cleaner_func,
                          save_path='temp_file.csv', download_override=False)

        return df

    def test_parse_hsdn(self):
        """Test ``dsi._harvest()`` for hsdn.
        Note: partially replicates ``dsi.hsdn_pull``"""
        hsdn_df = self._pickle_to_tsv_to_df('hsdn', cleaner_func=dsi._hsdn_df_cleaner)
        self.assertEqual(isinstance(hsdn_df, pd.DataFrame), True)
        self.assertEqual(hsdn_df.shape[0] > 0, True)

    def test_parse_rephetio_ml(self):
        """Test ``dsi._harvest()`` for rephetio_ml.
        Note: partially replicates ``dsi.rephetio_ml_pull``"""
        rephetio_ml_df = self._pickle_to_tsv_to_df('rephetio_ml', cleaner_func=dsi._rephetio_ml_df_cleaner)
        self.assertEqual(isinstance(rephetio_ml_df, pd.DataFrame), True)
        self.assertEqual(rephetio_ml_df.shape[0] > 0, True)

    def test_do_combine(self):
        """Test ``dsi._combine``."""
        hsdn_df = self._pickle_to_tsv_to_df('hsdn', cleaner_func=dsi._hsdn_df_cleaner)
        rephetio_ml_df = self._pickle_to_tsv_to_df('rephetio_ml', cleaner_func=dsi._rephetio_ml_df_cleaner)

        combined_df = dsi._combine(download_override=False, hsdn=hsdn_df, rephetio=rephetio_ml_df)
        self.assertEqual(isinstance(combined_df, pd.DataFrame), True)


unittest.main(exit=False)


# Note: clean up is redundant on Travis-CI
if "/home/travis/" not in os.getcwd():
    for f in ('hsdn', 'rephetio_ml'):
        os.remove(os_join(os.getcwd(), "data/{0}.tsv".format(f)))


sys.exit()
