"""
03_onnx_export/verify_onnx.py

Verify exported ONNX models run correctly and produce audio output.
Run this before handing off to Yiğit.

Usage:
    python verify_onnx.py --deliverable_dir ../06_deliverable/ --output_wav test_output.wav
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


# ---------------------------------------------------------------------------
# Minimal Turkish tokenizer (mirrors turkish_cleaners.py logic)
# ---------------------------------------------------------------------------

_TR_LOWER = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                           "abcçdefgğhıijklmnoöprsştuüvyz")


def text_to_ids(text: str, char_to_id: dict, eos_id: int) -> list:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TR_LOWER)
    text = re.sub(r"\s+", " ", text).strip()
    ids = [char_to_id.get(c, 0) for c in text]
    ids.append(eos_id)
    return ids


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def run_tts(text: str, deliverable_dir: Path) -> np.ndarray:
    # Load configs
    tok_cfg = json.loads((deliverable_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    aud_cfg = json.loads((deliverable_dir / "audio_config.json").read_text(encoding="utf-8"))

    char_to_id = {c: i for i, c in enumerate(tok_cfg["characters"])}
    eos_id = len(tok_cfg["characters"])

    token_ids = text_to_ids(text, char_to_id, eos_id)
    x = np.array([token_ids], dtype=np.int64)
    x_lengths = np.array([len(token_ids)], dtype=np.int64)

    # Session options
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Encoder
    enc_path = str(deliverable_dir / "vits_encoder.onnx")
    enc_sess = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
    enc_out = enc_sess.run(None, {"token_ids": x, "token_lengths": x_lengths})
    z_p = enc_out[0]  # [1, C, T_mel]

    # Vocoder
    voc_path = str(deliverable_dir / "vits_vocoder.onnx")
    voc_sess = ort.InferenceSession(voc_path, so, providers=["CPUExecutionProvider"])
    voc_out = voc_sess.run(None, {"z": z_p})
    audio = voc_out[0].squeeze()  # [T_wav]

    return audio, aud_cfg["sample_rate"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deliverable_dir", default="../06_deliverable/")
    parser.add_argument("--text", default="Merhaba, nasılsınız?")
    parser.add_argument("--output_wav", default="verify_output.wav")
    args = parser.parse_args()

    deliverable_dir = Path(args.deliverable_dir)

    print(f"Input text : {args.text}")
    print(f"ONNX dir   : {deliverable_dir}")

    audio, sr = run_tts(args.text, deliverable_dir)

    sf.write(args.output_wav, audio, sr)
    duration = len(audio) / sr
    print(f"Output wav : {args.output_wav}  ({duration:.2f}s, {sr} Hz)")

    # Latency estimate (rough — on CPU)
    import time
    runs = 5
    TARGET_WORD = "evet"
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_tts(TARGET_WORD, deliverable_dir)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    print(f"\nLatency ({runs} runs, single word '{TARGET_WORD}'): {avg_ms:.0f} ms avg")
    if avg_ms < 500:
        print("[OK] Latency target met (<500 ms on CPU).")
    else:
        print("[WARN] Latency exceeds 500 ms — consider INT8 quantization or smaller vocoder.")

    print("\nVerification complete. Hand off to Yiğit.")


if __name__ == "__main__":
    main()
