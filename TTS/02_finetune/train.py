"""
02_finetune/train.py

Fine-tune VITS on Turkish single-speaker data using coqui-tts.

Install deps first:
    pip install TTS torch torchaudio tensorboard

Usage:
    # First run (from pretrained baseline):
    python train.py --config vits_tr_finetune.json

    # Resume from checkpoint:
    python train.py --config vits_tr_finetune.json --restore_path runs/vits_tr_finetune/best_model.pth

    # List available Turkish pretrained models:
    python train.py --list_models

    # Download best available Turkish model as starting baseline:
    python train.py --download_baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path


def list_models():
    from TTS.api import TTS
    models = TTS.list_models()
    print("\n=== Available TTS models ===")
    for m in models:
        if "tr" in m.lower() or "turkish" in m.lower():
            print(f"  [TR] {m}")
    print("\nAll VITS models:")
    for m in models:
        if "vits" in m.lower():
            print(f"  {m}")


def download_baseline(out_dir: str = "pretrained_baseline"):
    from TTS.api import TTS
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Best available Turkish model — swap if a VITS variant appears
    MODEL_NAME = "tts_models/tr/common-voice/glow-tts"
    print(f"Downloading: {MODEL_NAME}")
    tts = TTS(MODEL_NAME)
    print(f"Model files cached to: {tts.manager.output_prefix}")
    return tts


def verify_data(data_path: str):
    """Quick sanity check on dataset before training."""
    meta = Path(data_path) / "metadata.csv"
    wavs = Path(data_path) / "wavs"
    if not meta.exists():
        print(f"[ERROR] metadata.csv not found at {meta}")
        sys.exit(1)
    if not wavs.is_dir():
        print(f"[ERROR] wavs/ folder not found at {wavs}")
        sys.exit(1)

    lines = meta.read_text(encoding="utf-8").splitlines()
    print(f"Metadata entries: {len(lines)}")
    missing = 0
    for line in lines:
        parts = line.split("|")
        wav_file = wavs / (parts[0] + ".wav")
        if not wav_file.exists():
            missing += 1
    if missing:
        print(f"[WARN] {missing} WAV files referenced in metadata.csv are missing!")
    else:
        print("[OK] All WAV files present.")

    # Estimate total duration
    try:
        import soundfile as sf
        total_sec = 0.0
        for wf in wavs.glob("*.wav"):
            info = sf.info(str(wf))
            total_sec += info.duration
        print(f"Total audio: {total_sec/3600:.2f} hours ({total_sec:.0f}s)")
        if total_sec < 3600:
            print("[WARN] Less than 1 hour of audio — fine-tune quality may be limited.")
    except ImportError:
        pass


def train(config_path: str, restore_path: str = None):
    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits, VitsAudioConfig
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    # Load config
    config = VitsConfig()
    config.load_json(config_path)

    # Override restore path if given on CLI
    if restore_path:
        config.restore_path = restore_path

    print(f"Training config: {config_path}")
    print(f"Restore from:    {config.restore_path or 'scratch'}")
    print(f"Output dir:      {config.output_path}")

    # Verify data first
    verify_data(config.datasets[0].path)

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(restore_path=config.restore_path, skip_train_epoch=False),
        config,
        output_path=config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vits_tr_finetune.json")
    parser.add_argument("--restore_path", default=None)
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--download_baseline", action="store_true")
    parser.add_argument("--verify_data_only", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if args.download_baseline:
        download_baseline()
        return

    if args.verify_data_only:
        cfg = json.load(open(args.config, encoding="utf-8"))
        verify_data(cfg["datasets"][0]["path"])
        return

    train(args.config, args.restore_path)


if __name__ == "__main__":
    main()
