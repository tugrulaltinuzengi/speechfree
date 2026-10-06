"""
06_deliverable/usage_example.py

Usage example for Yiğit — shows exactly how to call the ONNX TTS pipeline
from Python (mirrors what Android ONNX Runtime will do step-by-step).

Drop this file next to the ONNX files and run:
    pip install onnxruntime soundfile numpy
    python usage_example.py

Or call synthesize_word() from your own code.
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths — all files live in the same folder as this script
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
ENCODER_ONNX  = _HERE / "vits_encoder.onnx"
VOCODER_ONNX  = _HERE / "vits_vocoder.onnx"
TOKENIZER_CFG = _HERE / "tokenizer_config.json"
AUDIO_CFG     = _HERE / "audio_config.json"

# ---------------------------------------------------------------------------
# One-time setup — load configs and ONNX sessions
# ---------------------------------------------------------------------------
_tok_cfg  = json.loads(TOKENIZER_CFG.read_text(encoding="utf-8"))
_aud_cfg  = json.loads(AUDIO_CFG.read_text(encoding="utf-8"))

SAMPLE_RATE  = _aud_cfg["sample_rate"]      # 22050
CHAR_TO_ID   = {c: i for i, c in enumerate(_tok_cfg["characters"])}
EOS_ID       = len(_tok_cfg["characters"])

_so = ort.SessionOptions()
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

_enc_sess = ort.InferenceSession(str(ENCODER_ONNX), _so, providers=["CPUExecutionProvider"])
_voc_sess = ort.InferenceSession(str(VOCODER_ONNX), _so, providers=["CPUExecutionProvider"])

# Turkish locale-aware lowercase (critical — "I" → "ı", "İ" → "i")
_TR_LOWER = str.maketrans(
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
    "abcçdefgğhıijklmnoöprsştuüvyz",
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def text_to_tokens(text: str) -> np.ndarray:
    """Convert a Turkish string to a (1, T) int64 token array."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TR_LOWER)
    text = re.sub(r"\s+", " ", text).strip()
    ids = [CHAR_TO_ID.get(c, 0) for c in text] + [EOS_ID]
    return np.array([ids], dtype=np.int64)


def synthesize_word(text: str) -> np.ndarray:
    """
    Synthesize Turkish text → float32 audio array at SAMPLE_RATE Hz.

    Android ONNX Runtime equivalent:
        val tokens   = textToTokens(text)          // int64[1, T]
        val tokenLen = intArrayOf(tokens[0].size)  // int64[1]
        val zP       = encoderSession.run(...)      // float32[1, C, T_mel]
        val audio    = vocoderSession.run(...)      // float32[1, 1, T_wav]
    """
    x     = text_to_tokens(text)                          # [1, T]
    x_len = np.array([x.shape[1]], dtype=np.int64)       # [1]

    # Step 1 — Encoder: text tokens → latent representation
    z_p = _enc_sess.run(
        None,
        {"token_ids": x, "token_lengths": x_len},
    )[0]                                                   # [1, C, T_mel]

    # Step 2 — Vocoder: latent → waveform
    audio = _voc_sess.run(
        None,
        {"z": z_p},
    )[0].squeeze()                                         # [T_wav]

    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, path: str):
    """Write float32 audio to a 16-bit WAV file."""
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Demo — run to verify the whole pipeline end-to-end
# ---------------------------------------------------------------------------

TARGET_WORDS = [
    "EVET", "HAYIR", "MERHABA", "TEŞEKKÜRLER", "SU",
    "YARDIM", "TAMAM", "DUR", "GEL", "GÜNAYDIN",
]

if __name__ == "__main__":
    import time

    print(f"ONNX TTS Demo — sample rate: {SAMPLE_RATE} Hz\n")

    for word in TARGET_WORDS:
        t0 = time.perf_counter()
        audio = synthesize_word(word)
        latency_ms = (time.perf_counter() - t0) * 1000

        out_path = f"{word}.wav"
        save_wav(audio, out_path)
        dur_ms = len(audio) / SAMPLE_RATE * 1000

        status = "OK" if latency_ms < 500 else "SLOW"
        print(f"[{status}] {word:15s}  audio={dur_ms:.0f}ms  latency={latency_ms:.0f}ms  → {out_path}")

    print("\nAll 10 words synthesized. Yiğit: check WAVs, then replicate in Android ONNX Runtime.")
    print(f"\nNote for Android integration:")
    print(f"  - ONNX Runtime version: 1.16+ recommended")
    print(f"  - Input  'token_ids'   : int64[1, seq_len]")
    print(f"  - Input  'token_lengths': int64[1]")
    print(f"  - Output 'z_p'         : float32[1, channels, mel_len]  (encoder)")
    print(f"  - Input  'z'           : float32[1, channels, mel_len]  (vocoder)")
    print(f"  - Output 'audio'       : float32[1, 1, wav_len]  → squeeze → play")
    print(f"  - Sample rate          : {SAMPLE_RATE} Hz")
