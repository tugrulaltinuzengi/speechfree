"""
Build dataset_recordings.npz from the kayitN_* recording format.
Compatible with train_base_model.py (produces X, y, session_id arrays).

Run from tugrul_paket/model_and_shi/:
    python build_dataset_recordings.py
    python build_dataset_recordings.py --out dataset_recordings.npz --threshold_k 2.5

Matching strategy: temporal order — VAD segment i is paired with REP marker i.
Mirrors the assumption from build_dataset.py (segments arrive in spoken order).
"""

import os
import re
import argparse

import numpy as np
import pandas as pd

from signal_processing import preprocess
from vad import VAD, VAD_CONFIG

FS = 500
CALIBRATION_SAMPLES = 13 * FS   # 6500 samples — matches vad_compare.py
ADC_MAX = 1023

LABEL_MAPPING = {
    "EVET": 0, "HAYIR": 1, "MERHABA": 2, "TESEKKURLER": 3, "SU": 4,
    "YARDIM": 5, "TAMAM": 6, "DUR": 7, "GEL": 8, "GUNAYDIN": 9,
}

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "..", "recordings")

_REP_RE = re.compile(r'^([A-Z]+)_REP\d+$')


def parse_rep_labels(markers_path):
    """Return ordered list of label_ids for every REP marker, sorted by time."""
    df = pd.read_csv(markers_path)
    df.columns = df.columns.str.strip()
    labels = []
    for _, row in df.sort_values('time_s').iterrows():
        m = _REP_RE.match(str(row['label']).strip())
        if m:
            word = m.group(1)
            lid = LABEL_MAPPING.get(word, -1)
            if lid != -1:
                labels.append(lid)
    return labels


def build_dataset(output_file="dataset_recordings.npz", threshold_k=3.0):
    config = {**VAD_CONFIG, 'threshold_k': threshold_k}
    X_all, y_all, sid_all = [], [], []

    for rec_num in range(1, 5):
        name         = f"kayit{rec_num}"
        sensors_path = os.path.join(RECORDINGS_DIR, f"{name}_ALL_SENSORS.csv")
        markers_path = os.path.join(RECORDINGS_DIR, f"{name}_MARKERS.csv")

        if not os.path.exists(sensors_path) or not os.path.exists(markers_path):
            print(f"[SKIP] {name}: files missing")
            continue

        print(f"\n--- {name} ---")

        df = pd.read_csv(sensors_path)
        df.columns = df.columns.str.strip()
        # Permute: CSV pzt1, pzt2, emg1 → code expects [emg1, pzt1, pzt2]
        raw = df[["emg1", "pzt1", "pzt2"]].values.astype(np.int16)

        # ADC range check — warn if data looks 12-bit
        if raw.max() > ADC_MAX:
            print(f"  [WARN] max ADC value {raw.max()} > {ADC_MAX}; "
                  "consider passing adc_max=4095 to preprocess()")

        processed = preprocess(raw, fs=FS, adc_max=ADC_MAX)

        vad = VAD(config)
        vad.calibrate(processed[:CALIBRATION_SAMPLES])
        segments = vad.process_offline(processed[CALIBRATION_SAMPLES:])

        expected_labels = parse_rep_labels(markers_path)

        n_seg, n_lbl = len(segments), len(expected_labels)
        print(f"  VAD: {n_seg} segments | Expected: {n_lbl} labels")

        if n_seg != n_lbl:
            print(f"  [WARN] count mismatch — using min({n_seg}, {n_lbl})")

        n = min(n_seg, n_lbl)
        for i in range(n):
            # seg.data: (1000, 3)  →  .T: (3, 1000) for Conv1d input
            X_all.append(segments[i].data.T)
            y_all.append(expected_labels[i])
            sid_all.append(rec_num)

        print(f"  Collected {n} segments for kayit{rec_num}")

    if not X_all:
        print("ERROR: no segments collected — check recordings directory and file names.")
        return

    X   = np.array(X_all,   dtype=np.float32)
    y   = np.array(y_all,   dtype=np.int64)
    sid = np.array(sid_all, dtype=np.int64)

    np.savez(output_file, X=X, y=y, session_id=sid)

    print(f"\nSaved: {output_file}")
    print(f"  X shape: {X.shape}  (samples, channels, time)")
    print(f"  y shape: {y.shape}")
    uniq, cnts = np.unique(y, return_counts=True)
    print("  Class counts:", dict(zip(uniq.tolist(), cnts.tolist())))
    print("\nNext step: python train_base_model.py "
          f"--npz {os.path.basename(output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',         default='dataset_recordings.npz',
                        help='Output .npz filename (default: dataset_recordings.npz)')
    parser.add_argument('--threshold_k', type=float, default=3.0,
                        help='VAD threshold multiplier (match vad_compare.py setting)')
    args = parser.parse_args()
    build_dataset(output_file=args.out, threshold_k=args.threshold_k)
