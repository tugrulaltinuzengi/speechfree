import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

from evaluation import get_within_session_splits, get_cross_session_splits


def extract_features(seg_3ch, fs=500):
    """seg_3ch: (3, 1000) -> ~38 boyutlu feature vec"""
    feats = []
    for i in range(3):
        ch = seg_3ch[i]
        rms = np.sqrt(np.mean(ch**2))
        mav = np.mean(np.abs(ch))
        zcr = np.sum(np.diff(np.sign(ch)) != 0) / len(ch)
        wl = np.sum(np.abs(np.diff(ch)))
        activity = np.var(ch)
        dx = np.diff(ch); ddx = np.diff(dx)
        mob = np.std(dx)/np.std(ch) if np.std(ch) > 0 else 0
        comp = (np.std(ddx)/np.std(dx))/mob if mob > 0 and np.std(dx) > 0 else 0
        f, p = signal.welch(ch, fs, nperseg=256)
        mf = np.sum(f*p)/np.sum(p) if np.sum(p) > 0 else 0
        bp = [np.sum(p[(f >= lo) & (f < hi)]) for lo, hi in [(5,20),(20,50),(50,100),(100,201)]]
        feats.extend([rms, mav, zcr, wl, activity, mob, comp, mf, *bp])
    feats.append(np.nan_to_num(np.corrcoef(seg_3ch[1], seg_3ch[2])[0, 1]))
    feats.append(np.nan_to_num(np.corrcoef(seg_3ch[0], seg_3ch[1])[0, 1]))
    return np.array(feats)


def train_eval_rf(Xtr, ytr, Xte, yte):
    Ftr = np.array([extract_features(s) for s in Xtr])
    Fte = np.array([extract_features(s) for s in Xte])
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    rf.fit(Ftr, ytr)
    pred = rf.predict(Fte)
    proba = rf.predict_proba(Fte)
    acc = accuracy_score(yte, pred)
    top3 = np.mean([yte[i] in np.argsort(proba[i])[-3:] for i in range(len(yte))])
    return rf, acc, top3


def run_within_session(npz="dataset_poc.npz"):
    d = np.load(npz)
    X, y = d['X'], d['y']
    print(f"Within-Session 5-Fold (RF): {len(X)} segments, {X.shape[1]}ch")
    accs, t3s = [], []
    for f, (tr, te) in enumerate(get_within_session_splits(X, y, 5)):
        _, a, t3 = train_eval_rf(X[tr], y[tr], X[te], y[te])
        print(f"  Fold {f+1}: Top1={a*100:.1f}%  Top3={t3*100:.1f}%")
        accs.append(a); t3s.append(t3)
    print(f"RF Within-Session: Top1={np.mean(accs)*100:.2f}%  Top3={np.mean(t3s)*100:.2f}%")
    return np.mean(accs), np.mean(t3s)


def run_cross_session(npz="dataset_poc.npz"):
    d = np.load(npz)
    X, y, sid = d['X'], d['y'], d['session_id']
    sessions = {f"s{s}": (X[sid == s], y[sid == s]) for s in np.unique(sid)}
    print(f"Cross-Session LOSO (RF): {len(sessions)} sessions")
    accs, t3s = [], []
    for sp in get_cross_session_splits(sessions):
        _, a, t3 = train_eval_rf(sp['train_X'], sp['train_y'], sp['test_X'], sp['test_y'])
        print(f"  Test={sp['test_session_name']}: Top1={a*100:.1f}%  Top3={t3*100:.1f}%")
        accs.append(a); t3s.append(t3)
    print(f"RF Cross-Session: Top1={np.mean(accs)*100:.2f}%  Top3={np.mean(t3s)*100:.2f}%")
    return np.mean(accs), np.mean(t3s)


if __name__ == "__main__":
    run_within_session()
    print()
    run_cross_session()
