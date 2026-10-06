"""
tester/test_evaluation.py

Tests for 04_evaluation/evaluate_quality.py

Uses synthetic audio fixtures (sine waves) so no real recordings needed.

Run:
    python -m pytest tester/test_evaluation.py -v
"""

import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04_evaluation"))


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_sine_wav(path, freq=440.0, duration=2.0, sr=22050, amplitude=0.5):
    n = int(sr * duration)
    samples = [int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sr))
               for i in range(n)]
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        for s in samples:
            f.write(struct.pack("<h", max(-32768, min(32767, s))))


def _write_noise_wav(path, duration=2.0, sr=22050):
    import random
    n = int(sr * duration)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        for _ in range(n):
            v = random.randint(-16000, 16000)
            f.write(struct.pack("<h", v))


# ---------------------------------------------------------------------------
# MCD tests
# ---------------------------------------------------------------------------

class TestMCD(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        try:
            from evaluate_quality import mcd, compute_mfcc
            self.mcd = mcd
            self.compute_mfcc = compute_mfcc
            import numpy as np
            import soundfile as sf
            self.np = np
            self.sf = sf
            self.available = True
        except ImportError:
            self.available = False

    def test_identical_signals_low_mcd(self):
        if not self.available:
            self.skipTest("soundfile/scipy not installed")
        p = os.path.join(self.tmp, "ref.wav")
        _write_sine_wav(p)
        wav, sr = self.sf.read(p)
        wav = wav.astype(self.np.float32)
        value = self.mcd(wav, wav, sr)
        self.assertLess(value, 2.0, "MCD of identical signal should be near 0")

    def test_different_signals_higher_mcd(self):
        if not self.available:
            self.skipTest("soundfile/scipy not installed")
        ref_p = os.path.join(self.tmp, "ref.wav")
        syn_p = os.path.join(self.tmp, "syn.wav")
        _write_sine_wav(ref_p, freq=440)
        _write_noise_wav(syn_p)
        ref_wav, sr = self.sf.read(ref_p)
        syn_wav, _ = self.sf.read(syn_p)
        value = self.mcd(ref_wav.astype(self.np.float32),
                         syn_wav.astype(self.np.float32), sr)
        self.assertGreater(value, 1.0)

    def test_mcd_returns_float(self):
        if not self.available:
            self.skipTest("soundfile/scipy not installed")
        p = os.path.join(self.tmp, "a.wav")
        _write_sine_wav(p)
        wav, sr = self.sf.read(p)
        result = self.mcd(wav.astype(self.np.float32), wav.astype(self.np.float32), sr)
        self.assertIsInstance(result, float)

    def test_evaluate_mcd_scans_folder(self):
        if not self.available:
            self.skipTest("soundfile/scipy not installed")
        from evaluate_quality import evaluate_mcd
        ref_dir = Path(self.tmp) / "ref"
        syn_dir = Path(self.tmp) / "syn"
        ref_dir.mkdir()
        syn_dir.mkdir()
        for name in ["utt_001", "utt_002"]:
            _write_sine_wav(str(ref_dir / f"{name}.wav"), freq=440)
            _write_sine_wav(str(syn_dir / f"{name}.wav"), freq=440)
        results = evaluate_mcd(ref_dir, syn_dir)
        self.assertEqual(len(results), 2)
        self.assertIn("utt_001", results)

    def test_evaluate_mcd_ignores_unmatched_files(self):
        if not self.available:
            self.skipTest("soundfile/scipy not installed")
        from evaluate_quality import evaluate_mcd
        ref_dir = Path(self.tmp) / "ref2"
        syn_dir = Path(self.tmp) / "syn2"
        ref_dir.mkdir()
        syn_dir.mkdir()
        _write_sine_wav(str(ref_dir / "utt_001.wav"))
        _write_sine_wav(str(syn_dir / "utt_999.wav"))  # different name
        results = evaluate_mcd(ref_dir, syn_dir)
        self.assertEqual(len(results), 0)


# ---------------------------------------------------------------------------
# MOS HTML generation tests
# ---------------------------------------------------------------------------

class TestMOSHTML(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        try:
            from evaluate_quality import generate_mos_html
            self.generate_mos_html = generate_mos_html
            self.available = True
        except ImportError:
            self.available = False

    def test_generates_html_file(self):
        if not self.available:
            self.skipTest("evaluate_quality not importable")
        synth_dir = Path(self.tmp) / "synth"
        synth_dir.mkdir()
        for name in ["EVET", "HAYIR", "TAMAM"]:
            _write_sine_wav(str(synth_dir / f"{name}.wav"))
        out_html = os.path.join(self.tmp, "mos.html")
        self.generate_mos_html(synth_dir, output_path=out_html)
        self.assertTrue(os.path.exists(out_html))

    def test_html_contains_audio_tags(self):
        if not self.available:
            self.skipTest("evaluate_quality not importable")
        synth_dir = Path(self.tmp) / "synth2"
        synth_dir.mkdir()
        _write_sine_wav(str(synth_dir / "EVET.wav"))
        out_html = os.path.join(self.tmp, "mos2.html")
        self.generate_mos_html(synth_dir, output_path=out_html)
        content = Path(out_html).read_text(encoding="utf-8")
        self.assertIn("<audio", content)

    def test_html_contains_all_10_words(self):
        if not self.available:
            self.skipTest("evaluate_quality not importable")
        synth_dir = Path(self.tmp) / "synth3"
        synth_dir.mkdir()
        words = ["EVET", "HAYIR", "MERHABA", "TESEKKURLER", "SU",
                 "YARDIM", "TAMAM", "DUR", "GEL", "GUNAYDIN"]
        for w in words:
            _write_sine_wav(str(synth_dir / f"{w}.wav"))
        out_html = os.path.join(self.tmp, "mos3.html")
        self.generate_mos_html(synth_dir, output_path=out_html)
        content = Path(out_html).read_text(encoding="utf-8")
        for w in words:
            self.assertIn(w, content, f"Word not in HTML: {w}")

    def test_html_has_score_radios(self):
        if not self.available:
            self.skipTest("evaluate_quality not importable")
        synth_dir = Path(self.tmp) / "synth4"
        synth_dir.mkdir()
        _write_sine_wav(str(synth_dir / "TAMAM.wav"))
        out_html = os.path.join(self.tmp, "mos4.html")
        self.generate_mos_html(synth_dir, output_path=out_html)
        content = Path(out_html).read_text(encoding="utf-8")
        # Should have radio buttons for scores 1-5
        for score in ["1", "2", "3", "4", "5"]:
            self.assertIn(f'value="{score}"', content)

    def test_empty_synth_dir_produces_valid_html(self):
        if not self.available:
            self.skipTest("evaluate_quality not importable")
        synth_dir = Path(self.tmp) / "empty_synth"
        synth_dir.mkdir()
        out_html = os.path.join(self.tmp, "mos_empty.html")
        self.generate_mos_html(synth_dir, output_path=out_html)
        content = Path(out_html).read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", content)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
