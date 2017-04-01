"""

    BioVida-Genomics Subpackage Unit Testing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import sys
import unittest

# Allow access to modules
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from biovida import genomics


class GenomicsTests(unittest.TestCase):
    """

    Unit Tests for the Genomics Subpackage.

    """

    def test(self):
        self.assertEqual(1, 1)


unittest.main()
