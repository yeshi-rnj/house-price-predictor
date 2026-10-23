"""Tests for parsing, cleaning and formatting."""
import math
import unittest

import pandas as pd

from app.features import clean_raw_dataframe, parse_bhk, parse_sqft
from app.formatting import format_inr, inr_in_words


class ParsingTests(unittest.TestCase):
    def test_parse_sqft(self):
        self.assertEqual(parse_sqft("1200"), 1200.0)
        self.assertEqual(parse_sqft("2100 - 2850"), 2475.0)
        self.assertTrue(math.isnan(parse_sqft("34.46Sq. Meter")))
        self.assertTrue(math.isnan(parse_sqft("abc")))
        self.assertTrue(math.isnan(parse_sqft(None)))

    def test_parse_bhk(self):
        self.assertEqual(parse_bhk("2 BHK"), 2.0)
        self.assertEqual(parse_bhk("4 Bedroom"), 4.0)
        self.assertTrue(math.isnan(parse_bhk("studio")))
        self.assertTrue(math.isnan(parse_bhk(None)))


class CleaningTests(unittest.TestCase):
    def raw(self, rows):
        cols = ["area_type", "location", "size", "total_sqft", "bath", "balcony", "price"]
        return pd.DataFrame(rows, columns=cols)

    def test_drops_expected_rows_and_counts_them(self):
        rows = [
            ["Built-up  Area", " Whitefield ", "2 BHK", "1200", 2, 1, 80.0],      # kept
            ["Built-up  Area", " Whitefield ", "2 BHK", "1200", 2, 1, 80.0],      # duplicate
            ["Built-up  Area", "Hebbal", "3 BHK", "2100 - 2850", 3, None, 150.0],  # kept (range, missing balcony)
            ["Built-up  Area", "Hebbal", "2 BHK", "34.46Sq. Meter", 2, 1, 90.0],   # unparseable area
            ["Built-up  Area", "Hebbal", "6 BHK", "900", 6, 1, 90.0],              # too little area per bedroom
            ["Built-up  Area", "Hebbal", "2 BHK", "1100", 7, 1, 90.0],             # too many baths
            ["Built-up  Area", "Hebbal", "2 BHK", "1100", 2, 1, None],             # missing price
        ]
        clean, report = clean_raw_dataframe(self.raw(rows))
        self.assertEqual(len(clean), 2)
        self.assertEqual(report["rows_raw"], 7)
        self.assertEqual(report["dropped"]["exact_duplicates"], 1)
        self.assertEqual(report["dropped"]["too_little_area_per_bedroom"], 1)
        self.assertEqual(report["dropped"]["too_many_bathrooms_for_bedrooms"], 1)
        self.assertIn("Whitefield", set(clean["location"]))  # whitespace stripped
        self.assertEqual(report["balcony_missing_kept_for_imputation"], 1)

    def test_missing_columns_raise_clear_error(self):
        with self.assertRaises(ValueError):
            clean_raw_dataframe(pd.DataFrame({"foo": [1]}))


class FormattingTests(unittest.TestCase):
    def test_format_inr(self):
        self.assertEqual(format_inr(7500000), "\u20b975,00,000")
        self.assertEqual(format_inr(999), "\u20b9999")
        self.assertEqual(format_inr(1000), "\u20b91,000")
        self.assertEqual(format_inr(123456789), "\u20b912,34,56,789")
        self.assertEqual(format_inr(100000), "\u20b91,00,000")

    def test_in_words(self):
        self.assertEqual(inr_in_words(7500000), "\u20b975 lakh")
        self.assertEqual(inr_in_words(25000000), "\u20b92.5 crore")
        self.assertEqual(inr_in_words(85000), "\u20b985,000")


if __name__ == "__main__":
    unittest.main()
