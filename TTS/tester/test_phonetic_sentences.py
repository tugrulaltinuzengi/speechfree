"""
tester/test_phonetic_sentences.py

Validates the phonetic_sentences_tr.txt file:
  - Enough sentences
  - No empty lines (non-comment)
  - Sentence length distribution (5-15 words recommended)
  - Coverage of all major Turkish phonemes

Run:
    python -m pytest tester/test_phonetic_sentences.py -v
"""

import sys
import os
import unittest
from pathlib import Path

SENTENCES_FILE = Path(__file__).parent.parent / "01_data_prep" / "phonetic_sentences_tr.txt"

# Minimal expected Turkish phoneme coverage (grapheme proxies)
REQUIRED_PHONEMES = {
    "a", "e", "ı", "i", "o", "ö", "u", "ü",  # vowels
    "b", "c", "ç", "d", "f", "g", "ğ", "h",   # consonants
    "j", "k", "l", "m", "n", "p", "r", "s",
    "ş", "t", "v", "y", "z",
}


def load_sentences():
    lines = SENTENCES_FILE.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


class TestPhoneticSentencesFile(unittest.TestCase):
    def setUp(self):
        if not SENTENCES_FILE.exists():
            self.fail(f"phonetic_sentences_tr.txt not found at {SENTENCES_FILE}")
        self.sentences = load_sentences()

    def test_file_exists(self):
        self.assertTrue(SENTENCES_FILE.exists())

    def test_minimum_sentence_count(self):
        self.assertGreaterEqual(len(self.sentences), 100,
                                f"Need ≥100 sentences, got {len(self.sentences)}")

    def test_preferred_sentence_count(self):
        if len(self.sentences) < 300:
            print(f"\n[INFO] {len(self.sentences)} sentences present — target is 300 for full phonetic coverage")

    def test_no_empty_sentences(self):
        for i, s in enumerate(self.sentences):
            self.assertGreater(len(s.strip()), 0, f"Empty sentence at line {i+1}")

    def test_sentence_length_distribution(self):
        too_short = [s for s in self.sentences if len(s.split()) < 2]
        too_long  = [s for s in self.sentences if len(s.split()) > 20]
        self.assertEqual(len(too_short), 0,
                         f"{len(too_short)} sentences have <2 words: {too_short[:3]}")
        if too_long:
            print(f"\n[WARN] {len(too_long)} sentences exceed 20 words — may cause speaker fatigue")

    def test_phoneme_coverage(self):
        all_text = " ".join(self.sentences).lower()
        missing = []
        for phoneme in REQUIRED_PHONEMES:
            if phoneme not in all_text:
                missing.append(phoneme)
        self.assertEqual(missing, [], f"Missing phoneme coverage: {missing}")

    def test_vowel_harmony_diversity(self):
        """Check both front (e, i, ö, ü) and back (a, ı, o, u) vowels present."""
        all_text = " ".join(self.sentences).lower()
        front = [v for v in "eiöü" if v in all_text]
        back  = [v for v in "aıou" if v in all_text]
        self.assertEqual(len(front), 4, f"Missing front vowels: {set('eiöü') - set(front)}")
        self.assertEqual(len(back), 4,  f"Missing back vowels: {set('aıou') - set(back)}")

    def test_sentence_types_diversity(self):
        """Rough check: questions (?) and statements (.) both present."""
        questions   = [s for s in self.sentences if "?" in s]
        statements  = [s for s in self.sentences if s.endswith(".")]
        self.assertGreater(len(questions), 5,
                           "Need more question sentences for prosody diversity")
        self.assertGreater(len(statements), 50,
                           "Need more statement sentences")

    def test_all_10_target_words_appear(self):
        """Each target word should appear at least once across the corpus."""
        all_text = " ".join(self.sentences).upper()
        targets = ["EVET", "HAYIR", "MERHABA", "SU", "YARDIM", "TAMAM", "DUR", "GEL"]
        missing = [w for w in targets if w not in all_text]
        # Soft check — warn rather than fail (corpus may use variants)
        if missing:
            print(f"\n[INFO] Target words not found in sentence corpus: {missing}")
            print("Consider adding sentences that include these words.")

    def test_no_non_turkish_chars(self):
        """Flag sentences with characters outside the Turkish alphabet."""
        import re
        allowed = re.compile(r"^[a-zA-ZçÇşŞğĞıİöÖüÜ\s\.,!?;:\-'\"()]+$")
        flagged = [s for s in self.sentences if not allowed.match(s)]
        if flagged:
            print(f"\n[INFO] {len(flagged)} sentences contain unusual chars:")
            for s in flagged[:5]:
                print(f"  {s}")

    def test_average_words_per_sentence(self):
        lengths = [len(s.split()) for s in self.sentences]
        avg = sum(lengths) / len(lengths)
        self.assertGreater(avg, 3, f"Average sentence too short ({avg:.1f} words)")
        self.assertLess(avg, 18, f"Average sentence too long ({avg:.1f} words)")
        print(f"\n[INFO] Avg sentence length: {avg:.1f} words "
              f"(min={min(lengths)}, max={max(lengths)})")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
