"""PoC verisinden marker-based 3-kanal dataset üretir."""
import os, glob, re
import pandas as pd
import numpy as np
from signal_processing import preprocess

LABEL_MAP = {"EVET":0,"HAYIR":1,"MERHABA":2,"TESEKKURLER":3,"SU":4,
             "YARDIM":5,"TAMAM":6,"DUR":7,"GEL":8,"GUNAYDIN":9}

FS = 500
WINDOW = 1000     # 2s
PREROLL = 100     # 200ms önce başla — bip + reaction time için

REP_RE = re.compile(r'^([A-Z]+)_REP\d+$')


def parse_markers(path):
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    out = []
    for _, r in df.iterrows():
        m = REP_RE.match(str(r['label']).strip())
        if not m: continue
        word = m.group(1)
        if word not in LABEL_MAP: continue
        out.append((LABEL_MAP[word], int(round(r['time_s'] * FS))))
    return out


def build_session(sensors_csv, markers_csv, sid):
    df = pd.read_csv(sensors_csv); df.columns = df.columns.str.strip()
    # PoC kolonları: pzt1, pzt2, emg1 -> yeniden sırala [emg, pzt1, pzt2]
    raw = df[['emg1', 'pzt1', 'pzt2']].values.astype(np.int16)
    proc = preprocess(raw, fs=FS, adc_max=1023)
    # Per-session z-norm
    proc = (proc - proc.mean(0, keepdims=True)) / (proc.std(0, keepdims=True) + 1e-8)

    markers = parse_markers(markers_csv)
    X, y = [], []
    for label, idx in markers:
        s, e = idx - PREROLL, idx - PREROLL + WINDOW
        if s < 0: continue
        if e > len(proc):
            seg = np.vstack([proc[s:], np.zeros((e - len(proc), 3), dtype=proc.dtype)])
        else:
            seg = proc[s:e]
        X.append(seg.T.astype(np.float32))
        y.append(label)
    print(f"  {os.path.basename(sensors_csv)}: {len(X)} segments")
    return np.array(X), np.array(y, dtype=np.int64), np.full(len(X), sid, dtype=np.int64)


def build(data_dir="../poc_data", out="dataset_poc.npz"):
    files = sorted(glob.glob(os.path.join(data_dir, "kayit*_ALL_SENSORS.csv")))
    print(f"{len(files)} session files\n")
    Xs, ys, sids = [], [], []
    for i, sf in enumerate(files):
        mf = sf.replace('_ALL_SENSORS.csv', '_MARKERS.csv')
        if not os.path.exists(mf): continue
        X, y, s = build_session(sf, mf, sid=i)
        Xs.append(X); ys.append(y); sids.append(s)
    X = np.concatenate(Xs); y = np.concatenate(ys); sid = np.concatenate(sids)
    print(f"\nTOTAL: X={X.shape}  y={y.shape}  per-class={np.bincount(y)}  per-session={np.bincount(sid)}")
    np.savez(out, X=X, y=y, session_id=sid)
    print(f"Saved {out}")


if __name__ == "__main__":
    build()
