"""
01_data_prep/create_metadata.py

Builds LJSpeech-style metadata.csv from:
  - A folder of processed WAVs
  - A folder of matching .txt transcript files  (same stem as WAV)
    OR  a single master transcript file with one line per WAV (ordered)

Output metadata.csv format:
    filename|raw_text|normalized_text

Usage:
    # With per-file transcripts:
    python create_metadata.py --wavs data/wavs/ --transcripts data/transcripts/ --output data/

    # With single transcript file (one line per WAV, alphabetical order):
    python create_metadata.py --wavs data/wavs/ --transcript_file all_sentences.txt --output data/
"""

import argparse
import re
import unicodedata
from pathlib import Path


TURKISH_LOWER_MAP = str.maketrans(
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
    "abcçdefgğhıijklmnoöprsştuüvyz",
)


def turkish_lower(text: str) -> str:
    return text.translate(TURKISH_LOWER_MAP)


def normalize_turkish(text: str) -> str:
    """Lowercase + remove punctuation + collapse whitespace."""
    text = turkish_lower(text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_transcripts_from_folder(transcript_dir: Path) -> dict:
    mapping = {}
    for txt_file in transcript_dir.glob("*.txt"):
        raw = txt_file.read_text(encoding="utf-8").strip()
        mapping[txt_file.stem] = raw
    return mapping


def load_transcripts_from_file(transcript_file: Path, wav_stems: list) -> dict:
    lines = transcript_file.read_text(encoding="utf-8").splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    if len(lines) != len(wav_stems):
        print(f"[WARN] Transcript lines ({len(lines)}) != WAV files ({len(wav_stems)})")
    return {stem: line for stem, line in zip(wav_stems, lines)}


def build_metadata(wavs_dir: Path, transcript_map: dict, output_dir: Path):
    rows = []
    missing = []

    for wav_path in sorted(wavs_dir.glob("*.wav")):
        stem = wav_path.stem
        if stem not in transcript_map:
            missing.append(stem)
            continue
        raw = transcript_map[stem]
        normalized = normalize_turkish(raw)
        rows.append(f"{stem}|{raw}|{normalized}")

    out_path = output_dir / "metadata.csv"
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"Written {len(rows)} entries to {out_path}")
    if missing:
        print(f"[WARN] {len(missing)} WAVs had no transcript: {missing[:5]} …")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavs", required=True)
    parser.add_argument("--transcripts", default=None, help="Folder of .txt files")
    parser.add_argument("--transcript_file", default=None, help="Single file, one line per WAV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    wavs_dir = Path(args.wavs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_stems = sorted(p.stem for p in wavs_dir.glob("*.wav"))

    if args.transcripts:
        tmap = load_transcripts_from_folder(Path(args.transcripts))
    elif args.transcript_file:
        tmap = load_transcripts_from_file(Path(args.transcript_file), wav_stems)
    else:
        raise ValueError("Provide --transcripts or --transcript_file")

    build_metadata(wavs_dir, tmap, output_dir)


if __name__ == "__main__":
    main()
