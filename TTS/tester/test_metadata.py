"""
tester/test_metadata.py

Tests for 01_data_prep/create_metadata.py

Run:
    python -m pytest tester/test_metadata.py -v
"""

import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_data_prep"))
from create_metadata import (
    turkish_lower,
    normalize_turkish,
    build_metadata,
    load_transcripts_from_folder,
    load_transcripts_from_file,
)


def _make_dummy_wav(path):
    sr = 22050
    n = sr
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        f.write(b"\x00" * n * 2)


class TestTurkishLower(unittest.TestCase):
    def test_I_to_ı(self):
        self.assertEqual(turkish_lower("I"), "ı")

    def test_İ_to_i(self):
        self.assertEqual(turkish_lower("İ"), "i")


class TestNormalizeTurkish(unittest.TestCase):
    def test_removes_punctuation(self):
        result = normalize_turkish("Merhaba, nasılsın?")
        self.assertNotIn(",", result)
        self.assertNotIn("?", result)

    def test_lowercases(self):
        result = normalize_turkish("EVET")
        self.assertEqual(result, "evet")

    def test_collapses_whitespace(self):
        result = normalize_turkish("evet   hayır")
        self.assertNotIn("  ", result)

    def test_strips(self):
        result = normalize_turkish("  tamam  ")
        self.assertEqual(result, result.strip())

    def test_turkish_chars_preserved(self):
        for ch in ["ç", "ş", "ğ", "ı", "ö", "ü"]:
            result = normalize_turkish(ch)
            self.assertIn(ch, result)


class TestLoadTranscriptsFromFolder(unittest.TestCase):
    def test_reads_txt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "utt_001.txt").write_text("Merhaba dünya", encoding="utf-8")
            (p / "utt_002.txt").write_text("Evet, tamam", encoding="utf-8")
            result = load_transcripts_from_folder(p)
            self.assertEqual(result["utt_001"], "Merhaba dünya")
            self.assertEqual(result["utt_002"], "Evet, tamam")

    def test_ignores_non_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "utt_001.txt").write_text("Evet", encoding="utf-8")
            (p / "readme.md").write_text("ignore me", encoding="utf-8")
            result = load_transcripts_from_folder(p)
            self.assertIn("utt_001", result)
            self.assertNotIn("readme", result)


class TestLoadTranscriptsFromFile(unittest.TestCase):
    def test_maps_stems_to_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "transcripts.txt"
            p.write_text("Merhaba\nEvet\nHayır\n", encoding="utf-8")
            stems = ["utt_001", "utt_002", "utt_003"]
            result = load_transcripts_from_file(p, stems)
            self.assertEqual(result["utt_001"], "Merhaba")
            self.assertEqual(result["utt_003"], "Hayır")

    def test_mismatch_count_warns_but_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("Line1\nLine2\n", encoding="utf-8")
            stems = ["a", "b", "c"]  # 3 stems, 2 lines
            result = load_transcripts_from_file(p, stems)
            self.assertEqual(len(result), 2)


class TestBuildMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wavs_dir = Path(self.tmp) / "wavs"
        self.wavs_dir.mkdir()
        self.out_dir = Path(self.tmp) / "out"
        self.out_dir.mkdir()

    def _make_wavs(self, names):
        for n in names:
            _make_dummy_wav(str(self.wavs_dir / f"{n}.wav"))

    def test_creates_metadata_csv(self):
        self._make_wavs(["utt_001", "utt_002"])
        tmap = {"utt_001": "Merhaba", "utt_002": "Evet"}
        build_metadata(self.wavs_dir, tmap, self.out_dir)
        csv = self.out_dir / "metadata.csv"
        self.assertTrue(csv.exists())

    def test_csv_format_pipe_separated(self):
        self._make_wavs(["utt_001"])
        tmap = {"utt_001": "Merhaba, nasılsın?"}
        build_metadata(self.wavs_dir, tmap, self.out_dir)
        lines = (self.out_dir / "metadata.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        parts = lines[0].split("|")
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], "utt_001")
        self.assertEqual(parts[1], "Merhaba, nasılsın?")
        self.assertNotIn("?", parts[2])   # normalized

    def test_missing_transcript_skipped(self):
        self._make_wavs(["utt_001", "utt_002"])
        tmap = {"utt_001": "Evet"}   # utt_002 missing
        build_metadata(self.wavs_dir, tmap, self.out_dir)
        lines = (self.out_dir / "metadata.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len([l for l in lines if l.strip()]), 1)

    def test_10_target_words(self):
        words = ["EVET", "HAYIR", "MERHABA", "TEŞEKKÜRLER", "SU",
                 "YARDIM", "TAMAM", "DUR", "GEL", "GÜNAYDIN"]
        names = [f"word_{i:02d}" for i in range(len(words))]
        self._make_wavs(names)
        tmap = {n: w for n, w in zip(names, words)}
        build_metadata(self.wavs_dir, tmap, self.out_dir)
        lines = (self.out_dir / "metadata.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len([l for l in lines if l.strip()]), 10)

    def test_normalized_text_is_lowercase(self):
        self._make_wavs(["utt_001"])
        tmap = {"utt_001": "GÜNAYDIN"}
        build_metadata(self.wavs_dir, tmap, self.out_dir)
        lines = (self.out_dir / "metadata.csv").read_text(encoding="utf-8").splitlines()
        normalized = lines[0].split("|")[2]
        self.assertEqual(normalized, normalized.lower())


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
