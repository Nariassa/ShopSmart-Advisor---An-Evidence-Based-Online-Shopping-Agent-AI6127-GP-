import unittest
from src.ui.presentation import normalize_citation_labels

class CitationPresentationTests(unittest.TestCase):
    def test_only_existing_sources_are_normalized(self):
        self.assertEqual(normalize_citation_labels('Text [C1], [5], [99].', {'C1':{},'C5':{}}),'Text [C1], [C5], [99].')
    def test_non_reference_numbers_are_unchanged(self):
        self.assertEqual(normalize_citation_labels('Battery lasts 5 hours.',{}),'Battery lasts 5 hours.')
