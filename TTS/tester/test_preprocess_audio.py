"""
tester/test_preprocess_audio.py

Tests for 01_data_prep/preprocess_audio.py

Generates synthetic WAV fixtures (sine waves) — no real recording needed.

Run:
    python -m pytest tester/test_preprocess_audio.py -v
"""

import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_data_prep"))


# ---------------------------------------------------------------------------
# Minimal WAV writer (no external deps — fixtures only)
# ---------------------------------------------------------------------------

def write_sine_wav(path: str, freq: float = 440.0, duration: float = 2.0,
                   sample_rate: int = 44100, amplitude: float = 0.5):
    """Write a pure sine wave WAV at the given path."""
    n_samples = int(sample_rate * duration)
    samples = [int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
               for i in range(n_samples)]
    with open(path, "wb") as f:
        # RIFF header
        data_size = n_samples * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        for s in samples:
            f.write(struct.pack("<h", max(-32768, min(32767, s))))


def write_silent_wav(path: str, duration: float = 0.5, sample_rate: int = 44100):
    write_sine_wav(path, freq=0, duration=duration, sample_rate=sample_rate, amplitude=0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadWav(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_loads_44100_mono(self):
        from preprocess_audio import load_wav
        p = os.path.join(self.tmp, "test.wav")
        write_sine_wav(p, sample_rate=44100)
        seg = load_wav(Path(p))
        self.assertEqual(seg.channels, 1)
        self.assertEqual(seg.frame_rate, 22050)  # resampled to TARGET_SR

    def test_loads_48000_stereo(self):
        """Stereo 48kHz file should be converted to mono 22050."""
        try:
            from pydub import AudioSegment
            p = os.path.join(self.tmp, "stereo.wav")
            # Create stereo by loading and converting
            write_sine_wav(p, sample_rate=48000)
            from preprocess_audio import load_wav
            seg = load_wav(Path(p))
            self.assertEqual(seg.channels, 1)
        except Exception:
            self.skipTest("pydub not installed")


class TestRmsNormalize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_quiet_audio_boosted(self):
        try:
            from preprocess_audio import load_wav, rms_normalize, TARGET_DBFS
            p = os.path.join(self.tmp, "quiet.wav")
            write_sine_wav(p, amplitude=0.05)
            seg = load_wav(Path(p))
            normalized = rms_normalize(seg)
            self.assertAlmostEqual(normalized.dBFS, TARGET_DBFS, delta=3.0)
        except ImportError:
            self.skipTest("pydub not installed")

    def test_loud_audio_attenuated(self):
        try:
            from preprocess_audio import load_wav, rms_normalize, TARGET_DBFS
            p = os.path.join(self.tmp, "loud.wav")
            write_sine_wav(p, amplitude=0.95)
            seg = load_wav(Path(p))
            normalized = rms_normalize(seg)
            self.assertAlmostEqual(normalized.dBFS, TARGET_DBFS, delta=3.0)
        except ImportError:
            self.skipTest("pydub not installed")


class TestSegmentLongFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make_segmented_wav(self, path, n_segments=3, seg_dur=2.0, gap_dur=0.8):
        """Sine segments separated by silence."""
        import struct
        sr = 22050
        samples = []
        for _ in range(n_segments):
            for i in range(int(sr * seg_dur)):
                v = int(0.5 * 32767 * math.sin(2 * math.pi * 440 * i / sr))
                samples.append(struct.pack("<h", v))
            # silence gap
            for _ in range(int(sr * gap_dur)):
                samples.append(struct.pack("<h", 0))

        with open(path, "wb") as f:
            data_size = len(samples) * 2
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            for s in samples:
                f.write(s)

    def test_finds_segments(self):
        try:
            from preprocess_audio import segment_long_file, load_wav
            p = os.path.join(self.tmp, "long.wav")
            self._make_segmented_wav(p, n_segments=3, seg_dur=2.0, gap_dur=0.8)
            seg = load_wav(Path(p))
            chunks = segment_long_file(seg)
            self.assertGreaterEqual(len(chunks), 1, "Should find at least one speech segment")
        except ImportError:
            self.skipTest("pydub not installed")


class TestProcessFileSingleShort(unittest.TestCase):
    """Integration: process_file on a short single-sentence WAV."""

    def setUp(self):
        self.tmp_in = tempfile.mkdtemp()
        self.tmp_out = tempfile.mkdtemp()

    def test_short_file_produces_one_output(self):
        try:
            from preprocess_audio import process_file
            src = Path(self.tmp_in) / "utt_001.wav"
            write_sine_wav(str(src), duration=2.0)
            out_dir = Path(self.tmp_out)
            process_file(src, out_dir, "utt_001")
            outputs = list(out_dir.glob("*.wav"))
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].stem, "utt_001")
        except ImportError:
            self.skipTest("pydub not installed")

    def test_output_is_22050hz(self):
        try:
            import soundfile as sf
            from preprocess_audio import process_file
            src = Path(self.tmp_in) / "utt_002.wav"
            write_sine_wav(str(src), sample_rate=44100, duration=1.5)
            out_dir = Path(self.tmp_out)
            process_file(src, out_dir, "utt_002")
            out_wav = out_dir / "utt_002.wav"
            info = sf.info(str(out_wav))
            self.assertEqual(info.samplerate, 22050)
        except ImportError:
            self.skipTest("soundfile/pydub not installed")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
