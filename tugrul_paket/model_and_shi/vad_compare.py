"""
VAD vs Marker comparison pipeline — Steps 0-4 from DIRECTION_TUGRUL.md.

Run from tugrul_paket/model_and_shi/:
    python vad_compare.py                       # all 4 recordings, k=3.0
    python vad_compare.py --threshold_k 2.5    # tune threshold
    python vad_compare.py --recording kayit1   # single recording only

Outputs (../vad_compare_output/):
    kayitN_VAD_vs_MARKER.csv, summary.csv, per_recording_metrics.csv
    kayitN_timeline.png, delta_onset_histogram.png
    vad_compare_report.md
"""

import os
import re
import sys
import argparse

# Windows terminal may not support Unicode — force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import lfilter

from signal_processing import preprocess
from vad import VAD, VAD_CONFIG

# ── Constants ──────────────────────────────────────────────────────────────────

FS = 500
CALIBRATION_SAMPLES = 13 * FS        # 6500 samples — first REP ~14.3 s, 13 s safe
IOU_MATCH_THRESHOLD = 0.3
SNR_THRESHOLD_DB = 3.0
ADC_MAX = 1023                        # 10-bit recordings; use 4095 if hardware changed

LABEL_MAPPING = {
    "EVET": 0, "HAYIR": 1, "MERHABA": 2, "TESEKKURLER": 3, "SU": 4,
    "YARDIM": 5, "TAMAM": 6, "DUR": 7, "GEL": 8, "GUNAYDIN": 9,
}

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "..", "recordings")
OUTPUT_DIR     = os.path.join(_SCRIPT_DIR, "..", "vad_compare_output")

_REP_RE = re.compile(r'^([A-Z]+)_REP(\d+)$')
_END_RE = re.compile(r'^KELIME_([A-Z]+)_BITIS$')


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sensor_data(name):
    """Load kayitN_ALL_SENSORS.csv and permute columns to [emg1, pzt1, pzt2]."""
    path = os.path.join(RECORDINGS_DIR, f"{name}_ALL_SENSORS.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # CSV order: timestamp_s, pzt1, pzt2, emg1  →  code expects [emg1, pzt1, pzt2]
    return df[["emg1", "pzt1", "pzt2"]].values.astype(np.int16)


def parse_markers(name):
    """
    Return list of dicts for every REP marker:
        word, rep_num, label_id, onset_s, offset_s

    Offset rule:
      - Same word, next rep: next_onset − 200 ms
      - Last rep of word:    KELIME_X_BITIS time
    """
    path = os.path.join(RECORDINGS_DIR, f"{name}_MARKERS.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    word_end = {}
    for _, row in df.iterrows():
        m = _END_RE.match(str(row['label']).strip())
        if m:
            word_end[m.group(1)] = float(row['time_s'])

    reps = []
    for _, row in df.iterrows():
        m = _REP_RE.match(str(row['label']).strip())
        if m:
            reps.append((m.group(1), int(m.group(2)), float(row['time_s'])))
    reps.sort(key=lambda x: x[2])

    result = []
    for i, (word, rep_num, onset) in enumerate(reps):
        if i + 1 < len(reps) and reps[i + 1][0] == word:
            offset = reps[i + 1][2] - 0.2
        else:
            offset = word_end.get(word, onset + 2.0)
        if offset <= onset:
            offset = onset + 0.3
        result.append({
            'word': word,
            'rep_num': rep_num,
            'label_id': LABEL_MAPPING.get(word, -1),
            'onset_s': onset,
            'offset_s': offset,
        })
    return result


# ── VAD ────────────────────────────────────────────────────────────────────────

def run_vad(raw, threshold_k):
    config = {**VAD_CONFIG, 'threshold_k': threshold_k}
    processed = preprocess(raw, fs=FS, adc_max=ADC_MAX)
    vad = VAD(config)
    vad.calibrate(processed[:CALIBRATION_SAMPLES])
    segments = vad.process_offline(processed[CALIBRATION_SAMPLES:])
    return processed, vad, segments


def seg_times(seg):
    """Convert post-calibration segment indices to global time in seconds."""
    start_s = (CALIBRATION_SAMPLES + seg.start_idx) / FS
    end_s   = (CALIBRATION_SAMPLES + seg.end_idx)   / FS
    return start_s, end_s


# ── Metrics ────────────────────────────────────────────────────────────────────

def _iou(a0, a1, b0, b1):
    i0, i1 = max(a0, b0), min(a1, b1)
    if i1 <= i0:
        return 0.0
    return (i1 - i0) / (max(a1, b1) - min(a0, b0))


def greedy_match(vad_segs, refs):
    """
    Greedy IoU matching: highest IoU first, each segment matched at most once.
    Returns:
        pairs     — list of (ref_idx, vad_idx, iou_value)
        miss_idxs — ref indices with no VAD match
        fp_idxs   — VAD indices that matched no ref
    """
    n_r, n_v = len(refs), len(vad_segs)
    mat = np.zeros((n_r, n_v))
    for ri, ref in enumerate(refs):
        for vi, seg in enumerate(vad_segs):
            vs, ve = seg_times(seg)
            mat[ri, vi] = _iou(ref['onset_s'], ref['offset_s'], vs, ve)

    matched_r, matched_v = set(), set()
    pairs = []
    for flat in np.argsort(mat.ravel())[::-1]:
        ri, vi = np.unravel_index(flat, mat.shape)
        if mat[ri, vi] < IOU_MATCH_THRESHOLD:
            break
        if ri in matched_r or vi in matched_v:
            continue
        pairs.append((int(ri), int(vi), float(mat[ri, vi])))
        matched_r.add(ri)
        matched_v.add(vi)

    miss_idxs = [i for i in range(n_r) if i not in matched_r]
    fp_idxs   = [i for i in range(n_v) if i not in matched_v]
    return pairs, miss_idxs, fp_idxs


def compute_snr(processed, vad_segs):
    """sEMG RMS per VAD segment vs. calibration-period baseline."""
    baseline_rms = np.sqrt(np.mean(processed[:CALIBRATION_SAMPLES, 0] ** 2))
    if baseline_rms == 0:
        baseline_rms = 1e-9
    result = []
    for seg in vad_segs:
        i0 = CALIBRATION_SAMPLES + seg.start_idx
        i1 = CALIBRATION_SAMPLES + seg.end_idx
        emg = processed[i0:i1, 0]
        rms = np.sqrt(np.mean(emg ** 2)) if len(emg) > 0 else 0.0
        snr_db = 20.0 * np.log10(rms / baseline_rms) if rms > 0 else -np.inf
        result.append((rms, snr_db))
    return result, baseline_rms


# ── Per-recording pipeline ─────────────────────────────────────────────────────

def process_recording(name, threshold_k):
    print(f"\n{'='*52}\n{name}  (threshold_k={threshold_k})")

    raw  = load_sensor_data(name)
    refs = parse_markers(name)
    processed, vad, vad_segs = run_vad(raw, threshold_k)
    snr_info, _ = compute_snr(processed, vad_segs)

    print(f"  VAD: {len(vad_segs)} segments | Refs: {len(refs)}")

    pairs, miss_idxs, fp_idxs = greedy_match(vad_segs, refs)

    rows = []
    for ri, vi, iou_val in pairs:
        ref = refs[ri]
        vs, ve = seg_times(vad_segs[vi])
        rms, snr_db = snr_info[vi]
        rows.append({
            'rep_label':       f"{ref['word']}_REP{ref['rep_num']}",
            'word':            ref['word'],
            'rep_onset':       ref['onset_s'],
            'rep_offset':      ref['offset_s'],
            'vad_onset':       vs,
            'vad_offset':      ve,
            'delta_onset_ms':  (vs - ref['onset_s']) * 1000,
            'delta_offset_ms': (ve - ref['offset_s']) * 1000,
            'iou':             iou_val,
            'semg_rms':        rms,
            'snr_db':          snr_db,
            'low_snr':         snr_db < SNR_THRESHOLD_DB,
            'matched':         True,
        })

    for ri in miss_idxs:
        ref = refs[ri]
        rows.append({
            'rep_label':       f"{ref['word']}_REP{ref['rep_num']}",
            'word':            ref['word'],
            'rep_onset':       ref['onset_s'],
            'rep_offset':      ref['offset_s'],
            'vad_onset':       np.nan,
            'vad_offset':      np.nan,
            'delta_onset_ms':  np.nan,
            'delta_offset_ms': np.nan,
            'iou':             0.0,
            'semg_rms':        np.nan,
            'snr_db':          np.nan,
            'low_snr':         False,
            'matched':         False,
        })

    rows.sort(key=lambda r: r['rep_onset'])
    df = pd.DataFrame(rows)

    n_ref   = len(refs)
    n_match = len(pairs)
    ok      = df['matched']

    metrics = {
        'recording':            name,
        'n_ref':                n_ref,
        'n_vad':                len(vad_segs),
        'n_matched':            n_match,
        'match_rate':           n_match / n_ref if n_ref > 0 else 0.0,
        'miss_count':           len(miss_idxs),
        'false_positive_count': len(fp_idxs),
        'mean_delta_onset_ms':  df.loc[ok, 'delta_onset_ms'].mean(),
        'std_delta_onset_ms':   df.loc[ok, 'delta_onset_ms'].std(),
        'mae_delta_onset_ms':   df.loc[ok, 'delta_onset_ms'].abs().mean(),
        'mean_delta_offset_ms': df.loc[ok, 'delta_offset_ms'].mean(),
        'std_delta_offset_ms':  df.loc[ok, 'delta_offset_ms'].std(),
        'mae_delta_offset_ms':  df.loc[ok, 'delta_offset_ms'].abs().mean(),
        'mean_iou':             df.loc[ok, 'iou'].mean(),
        'threshold_k':          threshold_k,
        'vad_threshold':        vad.threshold,
    }

    print(f"  match={metrics['match_rate']:.1%}  FP={metrics['false_positive_count']}"
          f"  miss={metrics['miss_count']}  dOnset MAE={metrics['mae_delta_onset_ms']:.0f} ms")

    return df, metrics, processed, vad, vad_segs, refs


# ── Plots ──────────────────────────────────────────────────────────────────────

def _vad_envelope(processed, alpha):
    """Reconstruct the causal IIR envelope VAD uses, vectorized via lfilter."""
    rectified = np.abs(processed[:, 0])
    return lfilter([alpha], [1.0, -(1.0 - alpha)], rectified)


def plot_timeline(name, processed, vad, vad_segs, refs, out_dir):
    t        = np.arange(len(processed)) / FS
    envelope = _vad_envelope(processed, vad.alpha)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 5), sharex=True)

    ax1.plot(t, envelope, lw=0.4, color='steelblue', label='EMG envelope')
    ax1.axhline(vad.threshold, color='crimson', lw=1.0, linestyle='--',
                label=f'Threshold  (k={vad.config["threshold_k"]})')
    ax1.axvspan(0, CALIBRATION_SAMPLES / FS, alpha=0.08, color='gray',
                label='Calibration zone')
    ax1.set_ylabel('Amplitude')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_title(f'{name} — VAD vs Marker Timeline')

    ax2.set_ylim(0, 1)
    for seg in vad_segs:
        vs, ve = seg_times(seg)
        ax2.axvspan(vs, ve, alpha=0.35, color='green')
    for ref in refs:
        ax2.axvline(ref['onset_s'], color='crimson', lw=0.6, alpha=0.8)

    seen = set()
    for ref in sorted(refs, key=lambda r: r['onset_s']):
        if ref['word'] not in seen:
            ax2.text(ref['onset_s'], 0.65, ref['word'][:4],
                     fontsize=5, rotation=90, color='darkred', va='top')
            seen.add(ref['word'])

    ax2.legend(handles=[
        Patch(facecolor='green', alpha=0.4, label='VAD segments'),
        Line2D([0], [0], color='crimson', lw=0.8, label='REP onsets'),
    ], fontsize=7, loc='upper right')
    ax2.set_xlabel('Time (s)')
    ax2.set_yticks([])

    plt.tight_layout()
    path = os.path.join(out_dir, f'{name}_timeline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_delta_onset_hist(all_df, out_dir):
    deltas = all_df.loc[all_df['matched'], 'delta_onset_ms'].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(deltas, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.axvline(deltas.mean(), color='crimson', lw=1.5,
               label=f'Mean = {deltas.mean():.0f} ms')
    ax.axvline(deltas.median(), color='orange', lw=1.5, linestyle='--',
               label=f'Median = {deltas.median():.0f} ms')
    ax.set_xlabel('Δonset (ms)  [VAD onset − REP marker]')
    ax.set_ylabel('Count')
    ax.set_title(
        f'Onset error distribution  (n={len(deltas)},  MAE={deltas.abs().mean():.0f} ms)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, 'delta_onset_histogram.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ── Summary & report ───────────────────────────────────────────────────────────

def build_summary(all_df, all_metrics):
    matched  = all_df[all_df['matched']]
    word_grp = matched.groupby('word').agg(
        n_matched=('matched', 'count'),
        mean_delta_onset_ms=('delta_onset_ms', 'mean'),
        mae_delta_onset_ms=('delta_onset_ms', lambda x: x.abs().mean()),
        mean_iou=('iou', 'mean'),
    ).reset_index()
    n_ref_word = all_df.groupby('word').size().rename('n_ref').reset_index()
    summary = word_grp.merge(n_ref_word, on='word', how='right').fillna(0)
    summary['match_rate'] = summary['n_matched'] / summary['n_ref']
    summary['fp_total_all_recordings'] = sum(
        m['false_positive_count'] for m in all_metrics)
    return summary


def write_report(all_df, all_metrics, out_dir, threshold_k):
    matched    = all_df[all_df['matched']]
    overall_mr = all_df['matched'].mean()
    mae        = matched['delta_onset_ms'].abs().mean()
    mean_d     = matched['delta_onset_ms'].mean()
    std_d      = matched['delta_onset_ms'].std()
    mean_iou   = matched['iou'].mean()
    total_fp   = sum(m['false_positive_count'] for m in all_metrics)
    total_miss = sum(m['miss_count'] for m in all_metrics)
    low_snr_n  = int((all_df['low_snr'] == True).sum())

    lines = [
        "# VAD Comparison Report",
        "",
        f"**threshold_k = {threshold_k}**  |  "
        "Calibration = 13 s  |  Recordings: kayit1–kayit4  |  "
        f"IoU match threshold = {IOU_MATCH_THRESHOLD}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| match_rate | {overall_mr:.1%} ({all_df['matched'].sum()}/{len(all_df)}) |",
        f"| false_positive_count (total) | {total_fp} |",
        f"| miss_count (total) | {total_miss} |",
        f"| Δonset mean ± std | {mean_d:.0f} ± {std_d:.0f} ms |",
        f"| Δonset MAE | {mae:.0f} ms |",
        f"| mean IoU (matched pairs) | {mean_iou:.3f} |",
        f"| low_snr segments (<{SNR_THRESHOLD_DB} dB) | {low_snr_n} |",
        "",
        "## Per-recording",
        "",
        "| Recording | match_rate | FP | Miss | Δonset MAE (ms) | mean IoU |",
        "|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['recording']} | {m['match_rate']:.1%} "
            f"| {m['false_positive_count']} | {m['miss_count']} "
            f"| {m['mae_delta_onset_ms']:.0f} | {m['mean_iou']:.3f} |"
        )

    lines += [
        "",
        "## Threshold selection",
        "",
        f"Used `threshold_k={threshold_k}` (VAD_CONFIG default = 3.0).  "
        "Lower k (e.g. 2.5) increases sensitivity but may raise FP count "
        "in recordings where PZT bleeds into EMG.  "
        "Re-run with `--threshold_k 2.5` to compare.",
        "",
        "## Observed failure modes",
        "",
        "- **Merged reps**: fast consecutive repetitions where inter-rep silence "
        "< `silence_gap_samples` (200 ms) may be collapsed into a single VAD segment → miss.",
        "- **False positives**: non-speech muscle artifacts or PZT mechanical noise "
        "exceeding threshold → spurious VAD segments.",
        f"- **Low-SNR segments**: {low_snr_n} matched segments below {SNR_THRESHOLD_DB} dB "
        "flagged `low_snr=True` — classifier likely mis-classifies these.",
        "",
        "## Accuracy comparison",
        "",
        "| Setup | Within-session Top-1 | Top-3 | Cross-session Top-1 |",
        "|---|---|---|---|",
        "| Previous baseline | 67% | 87% | 52% |",
        "| New VAD segments  | _run train_base_model.py_ | — | — |",
        "",
        "_After running `build_dataset_recordings.py` → `train_base_model.py`, "
        "fill in results above._",
    ]

    path = os.path.join(out_dir, 'vad_compare_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold_k', type=float, default=3.0,
                        help='VAD threshold multiplier (default 3.0)')
    parser.add_argument('--recording', type=str, default=None,
                        help='Run single recording, e.g. kayit1 (default: all 4)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = [args.recording] if args.recording else [f"kayit{i}" for i in range(1, 5)]

    all_dfs, all_metrics = [], []
    for name in names:
        df, metrics, processed, vad, vad_segs, refs = process_recording(
            name, args.threshold_k)

        out_csv = os.path.join(OUTPUT_DIR, f'{name}_VAD_vs_MARKER.csv')
        df.to_csv(out_csv, index=False)
        print(f"  CSV → {out_csv}")

        plot_timeline(name, processed, vad, vad_segs, refs, OUTPUT_DIR)

        df['recording'] = name
        all_dfs.append(df)
        all_metrics.append(metrics)

    all_df = pd.concat(all_dfs, ignore_index=True)

    build_summary(all_df, all_metrics).to_csv(
        os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(OUTPUT_DIR, 'per_recording_metrics.csv'), index=False)

    print('\n--- Plots & report ---')
    plot_delta_onset_hist(all_df, OUTPUT_DIR)
    write_report(all_df, all_metrics, OUTPUT_DIR, args.threshold_k)

    print('\n=== DONE ===')
    matched = all_df[all_df['matched']]
    print(f"Overall match rate : {all_df['matched'].mean():.1%}")
    print(f"Δonset MAE         : {matched['delta_onset_ms'].abs().mean():.0f} ms")
    print(f"Mean IoU           : {matched['iou'].mean():.3f}")
    print(f"Total FP           : {sum(m['false_positive_count'] for m in all_metrics)}")
    print(f"Total misses       : {sum(m['miss_count'] for m in all_metrics)}")


if __name__ == '__main__':
    main()
