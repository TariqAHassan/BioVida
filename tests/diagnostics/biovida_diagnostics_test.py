"""

    BioVida-Diagnostics Subpackage Unit Testing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import sys
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

data_path = "tests/diagnostics/data"
obo_file = open(os_join(data_path, "obo_file.txt"), "r").read()


class DiseaseOntInterfaceTests(unittest.TestCase):
    """
    
    Unit Tests for the DiseaseOntInterface Class.
    
    """
    def test_parse(self):
        """Test the the data harvesting/processing engine generates a DataFrame."""
        doi._harvest_engine(disease_ontology_db_url='', obo_file=obo_file)
        self.assertEqual(isinstance(doi.disease_db, pd.DataFrame), True)

    def test_correct_dims_db(self):
        """Test that the dimensions of ``doi.disease_db`` are correct."""
        doi._harvest_engine(disease_ontology_db_url='', obo_file=obo_file)

        # Columns
        required_cols = ['alt_id', 'comment', 'created_by', 'creation_date', 'def',
                         'def_urls', 'disjoint_from', 'id', 'is_a', 'is_a_doid',
                         'is_obsolete', 'name', 'namespace', 'subset', 'synonym',
                         'synonym_exact', 'synonym_narrow', 'synonym_related', 'xref']
        n_missing_columns = len(set(required_cols) - set(doi.disease_db.columns))
        self.assertEqual(n_missing_columns == 0, True)

        # Rows
        n_terms = obo_file.count("[Term]")
        self.assertEqual(doi.disease_db.shape[0] == n_terms, True)


class DiseaseSymptomsInterfaceTests(unittest.TestCase):
    """

    Unit Tests for the DiseaseSymptomsInterface Class.

    """

    @staticmethod
    def _get_hsdn_df():
        hsdn_file_path = os_join(data_path, "hsdn.tsv")
        hsdn_df = dsi._harvest(url=hsdn_file_path,
                               cleaner_func=dsi._hsdn_df_cleaner,
                               save_path='hsdn.csv',
                               download_override=False)
        return hsdn_df

    def test_parse_hsdn(self):
        """Test ``dsi._harvest()`` for hsdn.
        Note: partially replicates ``dsi.hsdn_pull``"""
        hsdn_df = self._get_hsdn_df()
        self.assertEqual(isinstance(hsdn_df, pd.DataFrame), True)
        self.assertEqual(hsdn_df.shape[0] > 0, True)

    @staticmethod
    def _get_rephetio_ml_df():
        rephetio_ml_path = os_join(data_path, "rephetio_ml.tsv")
        rephetio_ml_df = dsi._harvest(url=rephetio_ml_path,
                                      cleaner_func=dsi._hsdn_df_cleaner,
                                      save_path='hsdn.csv',
                                      download_override=False)
        return rephetio_ml_df

    def test_parse_rephetio_ml(self):
        rephetio_ml_df = self._get_rephetio_ml_df()
        self.assertEqual(isinstance(rephetio_ml_df, pd.DataFrame), True)
        self.assertEqual(rephetio_ml_df.shape[0] > 0, True)

    def test_do_combine(self):
        """Test ``dsi._combine``."""
        combine_df = dsi._combine(download_override=False,
                                  hsdn=self._get_hsdn_df(),
                                  rephetio=self._get_rephetio_ml_df())
        self.assertEqual(isinstance(combine_df, pd.DataFrame), True)


unittest.main()
