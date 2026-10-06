"""
01_data_prep/preprocess_audio.py

Given a folder of raw WAV recordings (one long file or per-sentence files),
this script:
  1. Resamples to 22050 Hz mono (model target)
  2. Applies RMS normalization (-20 dBFS)
  3. Trims leading/trailing silence
  4. Optionally runs a noise gate using a noise profile sample
  5. Segments long files at silence boundaries → per-sentence WAVs

Usage:
    python preprocess_audio.py --input raw_audio/ --output data/wavs/ [--noise_sample noise.wav]
"""

import argparse
import os
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent


TARGET_SR = 22050
TARGET_DBFS = -20.0
MIN_SILENCE_MS = 500       # gap between sentences (ms)
SILENCE_THRESH_DB = -40    # dBFS below which is "silence"
MIN_CHUNK_MS = 1500        # drop chunks shorter than this (ms)
MAX_CHUNK_MS = 15000       # warn if chunk longer than this (ms)


def load_wav(path: Path) -> AudioSegment:
    audio = AudioSegment.from_file(str(path))
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
    return audio


def rms_normalize(audio: AudioSegment, target_dbfs: float = TARGET_DBFS) -> AudioSegment:
    delta = target_dbfs - audio.dBFS
    return audio.apply_gain(delta)


def noise_gate(audio: AudioSegment, noise_sample: AudioSegment, reduction_db: float = 15.0) -> AudioSegment:
    """Crude spectral subtraction proxy: attenuate frames below noise floor."""
    noise_dbfs = noise_sample.dBFS
    chunks = []
    frame_ms = 20
    for i in range(0, len(audio), frame_ms):
        frame = audio[i:i + frame_ms]
        if frame.dBFS < noise_dbfs + 6:
            frame = frame - reduction_db
        chunks.append(frame)
    return sum(chunks)


def segment_long_file(audio: AudioSegment) -> list:
    """Split a long recording on silence into per-sentence chunks."""
    chunks = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=100,
    )
    good = []
    for c in chunks:
        if len(c) < MIN_CHUNK_MS:
            continue
        if len(c) > MAX_CHUNK_MS:
            print(f"  [WARN] chunk {len(good)+1} is {len(c)/1000:.1f}s — may be two sentences")
        good.append(c)
    return good


def process_file(src: Path, out_dir: Path, base_name: str, noise_seg=None):
    print(f"Processing {src.name} …")
    audio = load_wav(src)

    if noise_seg:
        audio = noise_gate(audio, noise_seg)

    audio = rms_normalize(audio)

    if len(audio) > 30_000:
        # Long file → segment
        chunks = segment_long_file(audio)
        print(f"  → {len(chunks)} chunks found")
        for i, chunk in enumerate(chunks):
            out_path = out_dir / f"{base_name}_{i+1:03d}.wav"
            chunk.export(str(out_path), format="wav")
    else:
        # Short per-sentence file → trim & save
        non_silent = detect_nonsilent(audio, min_silence_len=200, silence_thresh=SILENCE_THRESH_DB)
        if non_silent:
            start, end = non_silent[0][0], non_silent[-1][1]
            audio = audio[max(0, start - 50): end + 50]
        out_path = out_dir / f"{base_name}.wav"
        audio.export(str(out_path), format="wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with raw WAV files")
    parser.add_argument("--output", required=True, help="Output folder for processed WAVs")
    parser.add_argument("--noise_sample", default=None, help="Short WAV of room noise for gate")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_seg = None
    if args.noise_sample:
        noise_seg = load_wav(Path(args.noise_sample))
        print(f"Noise profile loaded: {noise_seg.dBFS:.1f} dBFS")

    wav_files = sorted(in_dir.glob("*.wav")) + sorted(in_dir.glob("*.WAV"))
    if not wav_files:
        print("No WAV files found in input folder.")
        return

    for wav_path in wav_files:
        base = wav_path.stem
        process_file(wav_path, out_dir, base, noise_seg)

    print(f"\nDone. Output in: {out_dir}")
    output_files = list(out_dir.glob("*.wav"))
    print(f"Total output files: {len(output_files)}")


if __name__ == "__main__":
    main()
