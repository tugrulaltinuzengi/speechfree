"""
05_prerendered/synthesize_words.py

Plan B: Pre-render the 10 target words as WAV files using the fine-tuned model.
If ONNX export is solid, use this only for MOS evaluation samples.
If ONNX export fails, these WAVs become the Android assets (pre-rendered fallback).

Output: prerendered/EVET.wav, HAYIR.wav, … (filename = Turkish word, uppercase)

Usage:
    # Using coqui-tts (PyTorch — runs on your dev machine):
    python synthesize_words.py \
        --model_path ../runs/vits_tr_finetune/best_model.pth \
        --config_path ../runs/vits_tr_finetune/config.json \
        --output_dir prerendered/

    # Using ONNX (same pipeline Yiğit uses — validates before hand-off):
    python synthesize_words.py \
        --onnx_dir ../06_deliverable/ \
        --output_dir prerendered/

    # Batch: all tones/inflections (for MOS variety):
    python synthesize_words.py ... --multi_tone
"""

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# 10 target words + natural sentence variants for MOS evaluation
# ---------------------------------------------------------------------------

TARGET_WORDS = [
    "EVET",
    "HAYIR",
    "MERHABA",
    "TEŞEKKÜRLER",
    "SU",
    "YARDIM",
    "TAMAM",
    "DUR",
    "GEL",
    "GÜNAYDIN",
]

# Sentence variants to evaluate prosody naturalness
SENTENCE_VARIANTS = {
    "EVET":         ["Evet.", "Evet, tabii.", "Evet mi?"],
    "HAYIR":        ["Hayır.", "Hayır, değil.", "Hayır mı?"],
    "MERHABA":      ["Merhaba.", "Merhaba, nasılsınız?"],
    "TEŞEKKÜRLER":  ["Teşekkürler.", "Çok teşekkürler."],
    "SU":           ["Su.", "Su lütfen.", "Su içmek istiyorum."],
    "YARDIM":       ["Yardım.", "Yardım eder misiniz?", "Yardım lütfen."],
    "TAMAM":        ["Tamam.", "Tamam, anlıyorum.", "Tamam mı?"],
    "DUR":          ["Dur.", "Dur lütfen.", "Dur bir dakika."],
    "GEL":          ["Gel.", "Gel buraya lütfen.", "Gel, hadi gidelim."],
    "GÜNAYDIN":     ["Günaydın.", "Günaydın, nasılsınız?"],
}


# ---------------------------------------------------------------------------
# coqui-tts (PyTorch) synthesis
# ---------------------------------------------------------------------------

def synthesize_pytorch(model_path: str, config_path: str, texts: list, out_dir: Path):
    from TTS.api import TTS

    tts = TTS(model_path=model_path, config_path=config_path)

    for text in texts:
        # Safe filename: uppercase word part
        fname = re.sub(r"[^\w]", "_", text.strip().upper()) + ".wav"
        out_path = str(out_dir / fname)
        tts.tts_to_file(text=text, file_path=out_path)
        dur = sf.info(out_path).duration
        print(f"  [{dur:.2f}s] {out_path}")


# ---------------------------------------------------------------------------
# ONNX synthesis (mirrors verify_onnx.py — validates mobile pipeline)
# ---------------------------------------------------------------------------

_TR_LOWER = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                           "abcçdefgğhıijklmnoöprsştuüvyz")


def _text_to_ids(text: str, char_to_id: dict, eos_id: int) -> list:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TR_LOWER)
    text = re.sub(r"\s+", " ", text).strip()
    return [char_to_id.get(c, 0) for c in text] + [eos_id]


def synthesize_onnx(onnx_dir: str, texts: list, out_dir: Path):
    import onnxruntime as ort

    onnx_dir = Path(onnx_dir)
    tok_cfg = json.loads((onnx_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    aud_cfg = json.loads((onnx_dir / "audio_config.json").read_text(encoding="utf-8"))

    char_to_id = {c: i for i, c in enumerate(tok_cfg["characters"])}
    eos_id = len(tok_cfg["characters"])
    sr = aud_cfg["sample_rate"]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    enc_sess = ort.InferenceSession(str(onnx_dir / "vits_encoder.onnx"), so, providers=["CPUExecutionProvider"])
    voc_sess = ort.InferenceSession(str(onnx_dir / "vits_vocoder.onnx"), so, providers=["CPUExecutionProvider"])

    for text in texts:
        ids = _text_to_ids(text, char_to_id, eos_id)
        x = np.array([ids], dtype=np.int64)
        x_len = np.array([len(ids)], dtype=np.int64)

        z_p = enc_sess.run(None, {"token_ids": x, "token_lengths": x_len})[0]
        audio = voc_sess.run(None, {"z": z_p})[0].squeeze()

        fname = re.sub(r"[^\w]", "_", text.strip().upper()) + ".wav"
        out_path = out_dir / fname
        sf.write(str(out_path), audio, sr)
        print(f"  [{len(audio)/sr:.2f}s] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, help="coqui .pth checkpoint")
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--onnx_dir", default=None, help="ONNX deliverable folder")
    parser.add_argument("--output_dir", default="prerendered/")
    parser.add_argument("--multi_tone", action="store_true", help="Also render sentence variants")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build text list
    texts = list(TARGET_WORDS)
    if args.multi_tone:
        for variants in SENTENCE_VARIANTS.values():
            texts.extend(variants)

    texts = list(dict.fromkeys(texts))  # deduplicate while preserving order

    print(f"Synthesizing {len(texts)} utterances → {out_dir}")

    if args.onnx_dir:
        print("Mode: ONNX (mobile-equivalent pipeline)")
        synthesize_onnx(args.onnx_dir, texts, out_dir)
    elif args.model_path and args.config_path:
        print("Mode: coqui-tts (PyTorch)")
        synthesize_pytorch(args.model_path, args.config_path, texts, out_dir)
    else:
        parser.error("Provide either --onnx_dir or (--model_path + --config_path)")

    print(f"\nDone. {len(list(out_dir.glob('*.wav')))} WAV files in {out_dir}")
    print("Next step: run evaluate_quality.py with --synth_dir for MOS sheet.")


if __name__ == "__main__":
    main()
