"""
03_onnx_export/export_vits.py

Export a fine-tuned VITS model to ONNX for mobile deployment (Yiğit / Android ONNX Runtime).

The VITS model has two separable parts:
  1. Text encoder + duration predictor + flow  (encoder.onnx)
  2. HiFi-GAN vocoder / decoder                (vocoder.onnx)

Exporting them separately is more reliable than exporting the end-to-end model at once.
Yiğit calls encoder → get mel → call vocoder → get audio.

Usage:
    python export_vits.py \
        --model_path runs/vits_tr_finetune/best_model.pth \
        --config_path runs/vits_tr_finetune/config.json \
        --output_dir ../06_deliverable/

    # Quantize to INT8 (reduces size, may slightly reduce quality):
    python export_vits.py ... --quantize
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Wrapper modules — isolate the two ONNX graphs
# ---------------------------------------------------------------------------

class EncoderONNXWrapper(nn.Module):
    """
    Wraps: text tokens → (z, y_mask, latent)
    ONNX inputs:  x (token ids) [1, T_text]
    ONNX outputs: z_p [1, C, T_mel], y_mask [1, 1, T_mel], x_lengths [1]
    """
    def __init__(self, vits_model):
        super().__init__()
        self.model = vits_model

    def forward(self, x, x_lengths):
        return self.model.inference_encoder(x, x_lengths)


class VocoderONNXWrapper(nn.Module):
    """
    Wraps: latent z → waveform
    ONNX inputs:  z [1, C, T_mel]
    ONNX outputs: audio [1, 1, T_wav]
    """
    def __init__(self, vits_model):
        super().__init__()
        self.dec = vits_model.dec

    def forward(self, z):
        return self.dec(z)


# ---------------------------------------------------------------------------
# Core export function
# ---------------------------------------------------------------------------

def export_encoder(model, tokenizer, out_dir: Path, opset: int = 14):
    wrapper = EncoderONNXWrapper(model).eval()

    # Dummy input: short sentence token ids
    dummy_text = "merhaba nasılsın"
    token_ids = tokenizer.text_to_ids(dummy_text)
    x = torch.tensor([token_ids], dtype=torch.long)
    x_lengths = torch.tensor([len(token_ids)], dtype=torch.long)

    out_path = str(out_dir / "vits_encoder.onnx")
    torch.onnx.export(
        wrapper,
        (x, x_lengths),
        out_path,
        opset_version=opset,
        input_names=["token_ids", "token_lengths"],
        output_names=["z_p", "y_mask"],
        dynamic_axes={
            "token_ids":    {1: "seq_len"},
            "z_p":          {2: "mel_len"},
            "y_mask":       {2: "mel_len"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    print(f"Encoder exported: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return out_path


def export_vocoder(model, out_dir: Path, opset: int = 14):
    wrapper = VocoderONNXWrapper(model).eval()

    # Dummy mel-length latent
    C = model.dec.in_channels
    T_mel = 100
    dummy_z = torch.randn(1, C, T_mel)

    out_path = str(out_dir / "vits_vocoder.onnx")
    torch.onnx.export(
        wrapper,
        dummy_z,
        out_path,
        opset_version=opset,
        input_names=["z"],
        output_names=["audio"],
        dynamic_axes={
            "z":     {2: "mel_len"},
            "audio": {2: "wav_len"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    print(f"Vocoder exported: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return out_path


def save_tokenizer_config(tokenizer, config, out_dir: Path):
    data = {
        "characters": list(config.characters.characters),
        "pad": config.characters.pad,
        "eos": config.characters.eos,
        "bos": config.characters.bos,
        "blank": config.characters.blank,
        "use_phonemes": config.use_phonemes,
        "text_cleaner": config.text_cleaner,
        "char_to_id": tokenizer.char_to_id,
    }
    out_path = out_dir / "tokenizer_config.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tokenizer config saved: {out_path}")


def save_audio_config(config, out_dir: Path):
    data = {
        "sample_rate": config.audio.sample_rate,
        "hop_length": config.audio.hop_length,
        "win_length": config.audio.win_length,
        "fft_size": config.audio.fft_size,
        "num_mels": config.audio.num_mels,
        "mel_fmin": config.audio.mel_fmin,
        "mel_fmax": config.audio.mel_fmax,
    }
    out_path = out_dir / "audio_config.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Audio config saved: {out_path}")


def quantize_onnx(onnx_path: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
    orig_mb = os.path.getsize(onnx_path) / 1e6
    quant_mb = os.path.getsize(q_path) / 1e6
    print(f"Quantized {onnx_path}: {orig_mb:.1f} MB → {quant_mb:.1f} MB  ({q_path})")
    return q_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config_path", required=True, help="Path to config.json")
    parser.add_argument("--output_dir", default="../06_deliverable/")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path} …")
    config = VitsConfig()
    config.load_json(args.config_path)

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    model = Vits(config, ap, tokenizer, speaker_manager=None)
    cp = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(cp["model"])
    model.eval()

    print(f"\nExporting ONNX (opset {args.opset}) …")
    enc_path = export_encoder(model, tokenizer, out_dir, args.opset)
    voc_path = export_vocoder(model, out_dir, args.opset)

    if args.quantize:
        print("\nQuantizing to INT8 …")
        quantize_onnx(enc_path)
        quantize_onnx(voc_path)

    save_tokenizer_config(tokenizer, config, out_dir)
    save_audio_config(config, out_dir)

    # Save speaker embedding placeholder (single speaker = zeros / no embedding)
    spk_emb = np.zeros((1, 256), dtype=np.float32)
    np.save(str(out_dir / "speaker_embedding.npy"), spk_emb)
    print(f"Speaker embedding saved: {out_dir}/speaker_embedding.npy")

    total_mb = sum(
        os.path.getsize(p) / 1e6
        for p in out_dir.glob("*.onnx")
    )
    print(f"\nTotal ONNX size: {total_mb:.1f} MB  (target <30 MB)")
    if total_mb > 30:
        print("[WARN] Size exceeds 30 MB target — run with --quantize or switch to smaller vocoder.")


if __name__ == "__main__":
    main()
