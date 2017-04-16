"""

    BioVida-Images Subpackage Unit Testing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Note: 'medpix.png' is simply a blank images of same correct size.
# To test the OpeniImageProcessing() class, it will need to be replaced
# with an actual MedPix image.

import os
import sys
import unittest
import pandas as pd
from os.path import join as os_join


# Allow access to modules
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


from biovida import images
from biovida.support_tools.support_tools import items_null
from biovida.images._interface_support.openi.openi_text_processing import openi_raw_extract_and_clean


data_path = os_join(str(os.getcwd()).split("/tests")[0], "tests/images/data")
raw_openi_data_df = pd.read_pickle(os_join(data_path, "sample_records_raw.p"))


class OpeniInterfaceTests(unittest.TestCase):
    """

    Unit Tests for the Images Subpackage.

    """

    def test_cleaning_raw(self):
        """Test Extracting Features From Raw Open-i Data & Cleaning it."""
        cleaned_df = openi_raw_extract_and_clean(raw_openi_data_df, clinical_cases_only=False,
                                                 verbose=False, cache_path=data_path)

        # Tests for the newly generate columns
        expected_new_columns = ('diagnosis', 'imaging_modality_from_text', 'sex',
                                'illness_duration_years', 'modality_full', 'image_problems_from_text',
                                'parsed_abstract', 'image_id_short', 'age', 'ethnicity', 'image_plane')
        new_columns = set(cleaned_df.columns) - set(raw_openi_data_df.columns)

        # - Number of new columns
        self.assertEqual(len(new_columns) >= 11, True)

        # - Checks that least all `expected_new_columns` columns are in `new_columns`,
        #   However, this will not fail if additional columns are added.
        self.assertEqual(all(e in new_columns for e in expected_new_columns), True)

        # Test for only floats
        for c in ('illness_duration_years', 'age'):
            float_test = all(isinstance(i, float) for i in cleaned_df[c])
            self.assertEqual(float_test, True)

        # Test for only strings
        for c in ('diagnosis', 'imaging_modality_from_text', 'sex',
                  'modality_full', 'image_plane', 'image_id_short'):
            string_test = all(isinstance(i, str) or items_null(i) for i in cleaned_df[c])
            self.assertEqual(string_test, True)

        # Test for only dictionaries
        dict_test = all(isinstance(i, dict) or items_null(i) for i in cleaned_df['parsed_abstract'])
        self.assertEqual(dict_test, True)

        # Test for tuples
        tuple_test = all(isinstance(i, tuple) or items_null(i) for i in cleaned_df['image_problems_from_text'])
        self.assertEqual(tuple_test, True)


unittest.main()
