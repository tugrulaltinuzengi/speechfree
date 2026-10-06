"""
04_evaluation/evaluate_quality.py

Three-in-one evaluation script:
  1. MCD (Mel Cepstral Distortion) — objective, lower is better (target <7 dB)
  2. Speaker similarity — cosine similarity via Resemblyzer (target >0.85)
  3. MOS helper — generates an HTML listening sheet for subjective evaluation

Usage:
    # Full evaluation (needs reference WAVs and synthesized WAVs):
    python evaluate_quality.py \
        --ref_dir data/wavs/ \
        --synth_dir synth_output/ \
        --output_report report.txt

    # Generate MOS HTML sheet only:
    python evaluate_quality.py --mos_only --synth_dir synth_output/ --output_html mos_sheet.html
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# MCD (Mel Cepstral Distortion)
# ---------------------------------------------------------------------------

def compute_mfcc(wav: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Simplified MFCC via scipy — avoids librosa dependency."""
    from scipy.signal import spectrogram
    from scipy.fftpack import dct

    # Pre-emphasis
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])

    # Spectrogram
    win_len = int(0.025 * sr)  # 25ms
    hop_len = int(0.010 * sr)  # 10ms
    n_fft = 1 << (win_len - 1).bit_length()  # next power of 2 >= win_len
    f, t, Sxx = spectrogram(wav, fs=sr, nperseg=win_len, noverlap=win_len - hop_len, nfft=n_fft)

    # Mel filterbank
    n_mels = 40
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(fbank, Sxx)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    mfcc = dct(filter_banks, type=2, axis=0, norm="ortho")[:n_mfcc, :]
    return mfcc


def mcd(ref_wav: np.ndarray, synth_wav: np.ndarray, sr: int) -> float:
    ref_mfcc = compute_mfcc(ref_wav, sr)      # [13, T_ref]
    syn_mfcc = compute_mfcc(synth_wav, sr)    # [13, T_syn]

    T = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :T]
    syn_mfcc = syn_mfcc[:, :T]

    diff = ref_mfcc[1:] - syn_mfcc[1:]  # skip C0
    mcd_value = (10.0 / math.log(10.0)) * math.sqrt(2.0) * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
    return mcd_value


def evaluate_mcd(ref_dir: Path, synth_dir: Path) -> dict:
    results = {}
    for ref_path in sorted(ref_dir.glob("*.wav")):
        synth_path = synth_dir / ref_path.name
        if not synth_path.exists():
            continue
        ref_wav, sr = sf.read(str(ref_path))
        syn_wav, sr2 = sf.read(str(synth_path))
        if sr != sr2:
            from scipy.signal import resample
            syn_wav = resample(syn_wav, int(len(syn_wav) * sr / sr2))
        results[ref_path.stem] = mcd(ref_wav.astype(np.float32), syn_wav.astype(np.float32), sr)

    if results:
        values = list(results.values())
        print(f"MCD — mean: {np.mean(values):.2f} dB, min: {np.min(values):.2f}, max: {np.max(values):.2f}")
        print("  (target <7 dB; lower is better)")
    return results


# ---------------------------------------------------------------------------
# Speaker similarity via Resemblyzer
# ---------------------------------------------------------------------------

def evaluate_speaker_similarity(ref_dir: Path, synth_dir: Path) -> dict:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        print("[SKIP] resemblyzer not installed. Run: pip install resemblyzer")
        return {}

    encoder = VoiceEncoder()
    results = {}

    ref_wavs = sorted(ref_dir.glob("*.wav"))
    if not ref_wavs:
        return results

    # Compute reference speaker embedding (mean over all reference files)
    ref_embeddings = []
    for ref_path in ref_wavs:
        wav = preprocess_wav(str(ref_path))
        emb = encoder.embed_utterance(wav)
        ref_embeddings.append(emb)
    ref_emb = np.mean(ref_embeddings, axis=0)

    # Compare synthesized files
    for synth_path in sorted(synth_dir.glob("*.wav")):
        wav = preprocess_wav(str(synth_path))
        syn_emb = encoder.embed_utterance(wav)
        sim = float(np.dot(ref_emb, syn_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(syn_emb)))
        results[synth_path.stem] = sim

    if results:
        values = list(results.values())
        print(f"Speaker similarity — mean: {np.mean(values):.3f}, min: {np.min(values):.3f}")
        print("  (target >0.85)")

    # Save speaker embedding for deliverable package
    np.save("speaker_embedding.npy", ref_emb)
    print("Saved speaker_embedding.npy (use in deliverable package)")

    return results


# ---------------------------------------------------------------------------
# MOS HTML listening sheet
# ---------------------------------------------------------------------------

MOS_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head><meta charset="utf-8"><title>TTS MOS Değerlendirme</title>
<style>
body {{ font-family: Arial; max-width: 900px; margin: auto; padding: 20px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
th {{ background: #f0f0f0; }}
.score-radio input {{ margin: 0 4px; }}
</style></head>
<body>
<h2>TTS MOS Değerlendirme Formu</h2>
<p>Her ses için 1–5 arası puan verin: 5=Mükemmel, 4=İyi, 3=Orta, 2=Zayıf, 1=Kötü</p>
<p>Değerlendirici adı: <input type="text" id="rater_name" placeholder="Adınız"></p>
<form id="mos_form">
<table>
<tr><th>#</th><th>Ses</th><th>Metin</th><th>Puan (1-5)</th></tr>
{rows}
</table>
</form>
<br>
<button onclick="submitResults()">Sonuçları Göster</button>
<pre id="output"></pre>
<script>
function submitResults() {{
    const name = document.getElementById('rater_name').value;
    const scores = {{}};
    document.querySelectorAll('input[type=radio]:checked').forEach(r => {{
        scores[r.name] = parseInt(r.value);
    }});
    const result = {{ rater: name, scores: scores, timestamp: new Date().toISOString() }};
    document.getElementById('output').textContent = JSON.stringify(result, null, 2);
}}
</script>
</body></html>
"""

ROW_TEMPLATE = """<tr>
  <td>{idx}</td>
  <td><audio controls src="{audio_path}"></audio></td>
  <td>{text}</td>
  <td class="score-radio">
    {radios}
  </td>
</tr>"""


def generate_mos_html(synth_dir: Path, metadata_csv: Path = None, output_path: str = "mos_sheet.html"):
    synth_files = sorted(synth_dir.glob("*.wav"))

    text_map = {}
    if metadata_csv and metadata_csv.exists():
        for line in metadata_csv.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) >= 2:
                text_map[parts[0]] = parts[1]

    rows = []
    for idx, wav_path in enumerate(synth_files, 1):
        text = text_map.get(wav_path.stem, "—")
        radios = " ".join(
            f'<label>{s}<input type="radio" name="{wav_path.stem}" value="{s}"></label>'
            for s in [1, 2, 3, 4, 5]
        )
        rows.append(ROW_TEMPLATE.format(
            idx=idx,
            audio_path=str(wav_path),
            text=text,
            radios=radios,
        ))

    html = MOS_HTML_TEMPLATE.format(rows="\n".join(rows))
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"MOS sheet written: {output_path}")
    print(f"  Open in browser, share with ≥5 raters, target MOS ≥ 3.5")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", default=None)
    parser.add_argument("--synth_dir", required=True)
    parser.add_argument("--output_report", default="eval_report.json")
    parser.add_argument("--output_html", default="mos_sheet.html")
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--mos_only", action="store_true")
    args = parser.parse_args()

    synth_dir = Path(args.synth_dir)
    report = {}

    if not args.mos_only and args.ref_dir:
        ref_dir = Path(args.ref_dir)
        print("\n=== MCD Evaluation ===")
        report["mcd"] = evaluate_mcd(ref_dir, synth_dir)
        print("\n=== Speaker Similarity ===")
        report["speaker_similarity"] = evaluate_speaker_similarity(ref_dir, synth_dir)
        Path(args.output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull report: {args.output_report}")

    print("\n=== MOS HTML Sheet ===")
    meta = Path(args.metadata_csv) if args.metadata_csv else None
    generate_mos_html(synth_dir, meta, args.output_html)


if __name__ == "__main__":
    main()
