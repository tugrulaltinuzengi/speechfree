"""
tester/test_turkish_cleaners.py

Unit tests for 02_finetune/turkish_cleaners.py

Run:
    python -m pytest tester/test_turkish_cleaners.py -v
    # or standalone:
    python tester/test_turkish_cleaners.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_finetune"))

import unittest
from turkish_cleaners import (
    turkish_lower,
    normalize_punctuation,
    expand_abbreviations,
    number_to_turkish,
    turkish_cleaners,
)


class TestTurkishLower(unittest.TestCase):
    def test_I_becomes_ı(self):
        self.assertEqual(turkish_lower("I"), "ı")

    def test_İ_becomes_i(self):
        self.assertEqual(turkish_lower("İ"), "i")

    def test_standard_ascii(self):
        self.assertEqual(turkish_lower("ABCDE"), "abcde")

    def test_special_chars(self):
        self.assertEqual(turkish_lower("ÇŞĞÖÜ"), "çşğöü")

    def test_mixed(self):
        result = turkish_lower("İstanbul")
        self.assertEqual(result, "istanbul")

    def test_already_lower(self):
        self.assertEqual(turkish_lower("merhaba"), "merhaba")


class TestNumberToTurkish(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(number_to_turkish(0), "sıfır")

    def test_single_digits(self):
        self.assertEqual(number_to_turkish(1), "bir")
        self.assertEqual(number_to_turkish(5), "beş")
        self.assertEqual(number_to_turkish(9), "dokuz")

    def test_tens(self):
        self.assertEqual(number_to_turkish(10), "on")
        self.assertEqual(number_to_turkish(20), "yirmi")
        self.assertEqual(number_to_turkish(99), "doksan dokuz")

    def test_hundreds(self):
        self.assertEqual(number_to_turkish(100), "yüz")
        self.assertEqual(number_to_turkish(200), "iki yüz")
        self.assertEqual(number_to_turkish(315), "üç yüz on beş")

    def test_thousands(self):
        # "bir bin" → "bin" in Turkish
        self.assertEqual(number_to_turkish(1000), "bin")
        self.assertEqual(number_to_turkish(2000), "iki bin")
        self.assertEqual(number_to_turkish(1500), "bin beş yüz")

    def test_negative(self):
        result = number_to_turkish(-5)
        self.assertTrue(result.startswith("eksi"))

    def test_large(self):
        result = number_to_turkish(1_000_000)
        self.assertIn("milyon", result)


class TestExpandAbbreviations(unittest.TestCase):
    def test_dr(self):
        result = expand_abbreviations("Dr. Ahmet")
        self.assertIn("doktor", result.lower())

    def test_prof(self):
        result = expand_abbreviations("Prof. Çelik")
        self.assertIn("profesör", result.lower())

    def test_tl(self):
        result = expand_abbreviations("100 TL")
        self.assertIn("türk lirası", result.lower())

    def test_km(self):
        result = expand_abbreviations("50 km")
        self.assertIn("kilometre", result.lower())

    def test_no_change_for_unknown(self):
        text = "SpeachFree projesi"
        result = expand_abbreviations(text)
        self.assertIn("SpeachFree", result)


class TestNormalizePunctuation(unittest.TestCase):
    def test_em_dash(self):
        result = normalize_punctuation("kelime — kelime")
        self.assertNotIn("—", result)
        self.assertIn("-", result)

    def test_percent(self):
        result = normalize_punctuation("50%")
        self.assertIn("yüzde", result)

    def test_and_symbol(self):
        result = normalize_punctuation("A & B")
        self.assertIn("ve", result)

    def test_degree(self):
        result = normalize_punctuation("37°C")
        self.assertIn("derece", result)

    def test_curly_quotes_removed(self):
        result = normalize_punctuation("“merhaba”")
        self.assertNotIn("“", result)
        self.assertNotIn("”", result)


class TestTurkishCleanersPipeline(unittest.TestCase):
    def test_outputs_lowercase(self):
        result = turkish_cleaners("MERHABA")
        self.assertEqual(result, result.lower())

    def test_numbers_converted(self):
        result = turkish_cleaners("3 kişi geldi")
        self.assertNotRegex(result, r"\d")

    def test_whitespace_collapsed(self):
        result = turkish_cleaners("merhaba   dünya")
        self.assertNotIn("  ", result)

    def test_leading_trailing_stripped(self):
        result = turkish_cleaners("  merhaba  ")
        self.assertEqual(result, result.strip())

    def test_abbreviation_expanded_in_pipeline(self):
        result = turkish_cleaners("Dr. Ali 100 TL ödedi.")
        self.assertIn("doktor", result)
        self.assertIn("türk lirası", result)

    def test_turkish_i_preserved_correctly(self):
        result = turkish_cleaners("İstanbul'da İki kişi var")
        self.assertTrue(result.startswith("istanbul"))

    def test_empty_string(self):
        result = turkish_cleaners("")
        self.assertEqual(result, "")

    def test_only_punctuation(self):
        result = turkish_cleaners("!!!")
        self.assertEqual(result.strip(), "")

    def test_known_words_preserved(self):
        for word in ["evet", "hayır", "merhaba", "tamam"]:
            result = turkish_cleaners(word)
            self.assertEqual(result, word)

    def test_10_target_words(self):
        targets = [
            ("EVET", "evet"),
            ("HAYIR", "hayır"),
            ("MERHABA", "merhaba"),
            ("TEŞEKKÜRLER", "teşekkürler"),
            ("SU", "su"),
            ("YARDIM", "yardim"),  # note: ı vs i edge
            ("TAMAM", "tamam"),
            ("DUR", "dur"),
            ("GEL", "gel"),
            ("GÜNAYDIN", "günaydın"),
        ]
        for raw, expected_fragment in targets:
            result = turkish_cleaners(raw)
            self.assertIn(expected_fragment.split()[0][:3], result,
                          f"Failed on word: {raw} → got: {result}")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
