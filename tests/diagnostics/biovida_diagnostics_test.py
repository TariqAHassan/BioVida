"""

    BioVida-Diagnostics Subpackage Unit Testing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import sys
import unittest
import pandas as pd

# Allow access to modules
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from biovida.diagnostics import DiseaseOntInterface
from biovida.diagnostics import DiseaseSymptomsInterface


doi = DiseaseOntInterface(verbose=False)
obo_file = open("tests/diagnostics/data/obo_file.txt", "r").read()
# dsi = DiseaseSymptomsInterface(verbose=False)


class DiseaseOntInterfaceTests(unittest.TestCase):
    """
    
    Unit Tests for the DiseaseOntInterface Class.
    
    """
    def parse(self):
        """Test the the data harvesting/processing engine generates a DataFrame."""
        doi._harvest_engine(disease_ontology_db_url=None, obo_file=obo_file)
        self.assertEqual(isinstance(doi.disease_db, pd.DataFrame), True)

    def correct_cols_db(self):
        """Test that all columns have been generated."""
        required_cols = ['alt_id', 'comment', 'created_by', 'creation_date', 'def',
                         'def_urls', 'disjoint_from', 'id', 'is_a', 'is_a_doid',
                         'is_obsolete', 'name', 'namespace', 'subset', 'synonym',
                         'synonym_exact', 'synonym_narrow', 'synonym_related', 'xref']
        n_missing_columns = len(set(required_cols) - set(doi.disease_db.columns))
        self.assertEqual(n_missing_columns == 0, True)

    def correct_len_db(self):
        """Test that the DataFrame generated a row for each '[Term]' in `obo_file`."""
        n_terms = obo_file.count("[Term]")
        self.assertEqual(doi.disease_db.shape[0] == n_terms, True)



# class DiseaseSymptomsInterfaceTests(unittest.TestCase):
#     """
#
#     Unit Tests for the DiseaseSymptomsInterface Class.
#
#     """

















unittest.main()
