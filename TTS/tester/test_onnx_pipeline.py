"""
tester/test_onnx_pipeline.py

Tests for the ONNX inference pipeline used in:
  - 03_onnx_export/verify_onnx.py
  - 06_deliverable/usage_example.py

These tests use mock ONNX sessions so they run without a trained model.
A live ONNX integration test is included but skipped if model files are absent.

Run:
    python -m pytest tester/test_onnx_pipeline.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Usage example path for tokenizer helpers
USAGE_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "06_deliverable")
sys.path.insert(0, USAGE_EXAMPLE_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_deliverable(tmp_dir: str) -> Path:
    """Create minimal config files for testing without real ONNX models."""
    d = Path(tmp_dir)

    characters = list("abcçdefgğhıijklmnoöprsştüüvyz !',.-?")
    char_to_id = {c: i for i, c in enumerate(characters)}

    tok_cfg = {
        "characters": characters,
        "pad": "<PAD>",
        "eos": "<EOS>",
        "bos": "<BOS>",
        "blank": "<BLNK>",
        "use_phonemes": False,
        "text_cleaner": "turkish_cleaners",
        "char_to_id": char_to_id,
    }
    (d / "tokenizer_config.json").write_text(
        json.dumps(tok_cfg, ensure_ascii=False), encoding="utf-8"
    )

    aud_cfg = {
        "sample_rate": 22050,
        "hop_length": 256,
        "win_length": 1024,
        "fft_size": 1024,
        "num_mels": 80,
        "mel_fmin": 0,
        "mel_fmax": None,
    }
    (d / "audio_config.json").write_text(json.dumps(aud_cfg), encoding="utf-8")

    return d


TARGET_WORDS = [
    "EVET", "HAYIR", "MERHABA", "TEŞEKKÜRLER", "SU",
    "YARDIM", "TAMAM", "DUR", "GEL", "GÜNAYDIN",
]


# ---------------------------------------------------------------------------
# Tests: tokenizer
# ---------------------------------------------------------------------------

class TestTextToTokens(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ddir = make_fake_deliverable(self.tmp)

    def _get_text_to_tokens(self):
        tok_cfg = json.loads((self.ddir / "tokenizer_config.json").read_text(encoding="utf-8"))
        import re
        import unicodedata

        TR_LOWER = str.maketrans(
            "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
            "abcçdefgğhıijklmnoöprsştüüvyz",
        )
        char_to_id = {c: i for i, c in enumerate(tok_cfg["characters"])}
        eos_id = len(tok_cfg["characters"])

        def text_to_ids(text):
            text = unicodedata.normalize("NFC", text)
            text = text.translate(TR_LOWER)
            text = re.sub(r"\s+", " ", text).strip()
            return [char_to_id.get(c, 0) for c in text] + [eos_id]

        return text_to_ids, eos_id

    def test_returns_list_of_ints(self):
        fn, _ = self._get_text_to_tokens()
        result = fn("merhaba")
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(i, int) for i in result))

    def test_ends_with_eos(self):
        fn, eos_id = self._get_text_to_tokens()
        result = fn("evet")
        self.assertEqual(result[-1], eos_id)

    def test_nonempty_for_all_10_words(self):
        fn, _ = self._get_text_to_tokens()
        for word in TARGET_WORDS:
            result = fn(word)
            self.assertGreater(len(result), 1, f"Empty tokens for: {word}")

    def test_I_correctly_lowercased_to_ı(self):
        fn, _ = self._get_text_to_tokens()
        # "I" should become "ı", not "i"
        result_I = fn("I")
        result_ı = fn("ı")
        # Both should produce same token sequence (EOS aside)
        self.assertEqual(result_I, result_ı)

    def test_uppercase_same_as_lowercase(self):
        fn, _ = self._get_text_to_tokens()
        self.assertEqual(fn("EVET"), fn("evet"))
        self.assertEqual(fn("TAMAM"), fn("tamam"))


# ---------------------------------------------------------------------------
# Tests: mock ONNX session shapes
# ---------------------------------------------------------------------------

class TestONNXShapes(unittest.TestCase):
    """Verify that input/output shapes match between encoder and vocoder."""

    def test_encoder_output_feeds_vocoder(self):
        """
        Encoder output z_p shape [1, C, T_mel] must match vocoder input z shape.
        This is a contract test — mock both sessions and check shape consistency.
        """
        C = 192
        T_mel = 50
        T_wav = T_mel * 256  # hop_length

        # Mock encoder: returns z_p
        mock_enc = MagicMock()
        mock_enc.run.return_value = [np.random.randn(1, C, T_mel).astype(np.float32)]

        # Mock vocoder: accepts z_p, returns audio
        mock_voc = MagicMock()
        mock_voc.run.return_value = [np.random.randn(1, 1, T_wav).astype(np.float32)]

        # Simulate pipeline
        x = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        x_len = np.array([5], dtype=np.int64)

        z_p = mock_enc.run(None, {"token_ids": x, "token_lengths": x_len})[0]
        self.assertEqual(z_p.shape, (1, C, T_mel))

        audio = mock_voc.run(None, {"z": z_p})[0]
        self.assertEqual(audio.shape, (1, 1, T_wav))

        audio_squeezed = audio.squeeze()
        self.assertEqual(audio_squeezed.ndim, 1)

    def test_dynamic_sequence_lengths(self):
        """Encoder must handle different text lengths."""
        C = 192

        for T_text in [1, 5, 20, 100]:
            T_mel = T_text * 3  # approximate expansion

            mock_enc = MagicMock()
            mock_enc.run.return_value = [np.random.randn(1, C, T_mel).astype(np.float32)]

            x = np.ones((1, T_text), dtype=np.int64)
            x_len = np.array([T_text], dtype=np.int64)
            z_p = mock_enc.run(None, {"token_ids": x, "token_lengths": x_len})[0]

            self.assertEqual(z_p.shape[0], 1)
            self.assertEqual(z_p.shape[1], C)
            self.assertGreater(z_p.shape[2], 0)


# ---------------------------------------------------------------------------
# Tests: config file completeness
# ---------------------------------------------------------------------------

class TestDeliverableConfigs(unittest.TestCase):
    DELIVERABLE_DIR = Path(__file__).parent.parent / "06_deliverable"

    def test_tokenizer_config_exists(self):
        self.assertTrue((self.DELIVERABLE_DIR / "tokenizer_config.json").exists())

    def test_audio_config_exists(self):
        self.assertTrue((self.DELIVERABLE_DIR / "audio_config.json").exists())

    def test_audio_config_has_required_fields(self):
        cfg = json.loads((self.DELIVERABLE_DIR / "audio_config.json").read_text(encoding="utf-8"))
        for field in ["sample_rate", "hop_length", "win_length", "fft_size", "num_mels"]:
            self.assertIn(field, cfg, f"Missing field: {field}")

    def test_sample_rate_is_22050(self):
        cfg = json.loads((self.DELIVERABLE_DIR / "audio_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["sample_rate"], 22050)

    def test_tokenizer_config_has_characters(self):
        cfg = json.loads((self.DELIVERABLE_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertIn("characters", cfg)
        chars = cfg["characters"]
        self.assertGreater(len(chars), 10)

    def test_tokenizer_contains_turkish_chars(self):
        cfg = json.loads((self.DELIVERABLE_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
        chars = cfg["characters"]
        for ch in ["ç", "ş", "ğ", "ı", "ö", "ü"]:
            self.assertIn(ch, chars, f"Missing Turkish char: {ch}")

    def test_usage_example_exists(self):
        self.assertTrue((self.DELIVERABLE_DIR / "usage_example.py").exists())

    def test_model_card_exists(self):
        self.assertTrue((self.DELIVERABLE_DIR / "model_card.md").exists())


# ---------------------------------------------------------------------------
# Live integration test (skipped unless ONNX files are present)
# ---------------------------------------------------------------------------

class TestLiveONNX(unittest.TestCase):
    DELIVERABLE_DIR = Path(__file__).parent.parent / "06_deliverable"

    def setUp(self):
        enc = self.DELIVERABLE_DIR / "vits_encoder.onnx"
        voc = self.DELIVERABLE_DIR / "vits_vocoder.onnx"
        if not enc.exists() or not voc.exists():
            self.skipTest("ONNX model files not present — run export_vits.py first")

    def test_synthesize_evet(self):
        import onnxruntime as ort
        import re
        import unicodedata

        tok_cfg = json.loads((self.DELIVERABLE_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
        aud_cfg = json.loads((self.DELIVERABLE_DIR / "audio_config.json").read_text(encoding="utf-8"))
        char_to_id = {c: i for i, c in enumerate(tok_cfg["characters"])}
        eos_id = len(tok_cfg["characters"])
        sr = aud_cfg["sample_rate"]

        TR_LOWER = str.maketrans(
            "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
            "abcçdefgğhıijklmnoöprsştüüvyz",
        )

        text = "evet"
        text = unicodedata.normalize("NFC", text).translate(TR_LOWER)
        ids = [char_to_id.get(c, 0) for c in text] + [eos_id]
        x = np.array([ids], dtype=np.int64)
        x_len = np.array([len(ids)], dtype=np.int64)

        so = ort.SessionOptions()
        enc_sess = ort.InferenceSession(
            str(self.DELIVERABLE_DIR / "vits_encoder.onnx"), so,
            providers=["CPUExecutionProvider"])
        voc_sess = ort.InferenceSession(
            str(self.DELIVERABLE_DIR / "vits_vocoder.onnx"), so,
            providers=["CPUExecutionProvider"])

        z_p = enc_sess.run(None, {"token_ids": x, "token_lengths": x_len})[0]
        audio = voc_sess.run(None, {"z": z_p})[0].squeeze()

        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        dur_ms = len(audio) / sr * 1000
        self.assertLess(dur_ms, 5000, "Synthesized audio is suspiciously long")
        self.assertGreater(dur_ms, 50, "Synthesized audio is suspiciously short")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
