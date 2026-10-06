"""
Retrain + evaluate — Step 5 & confusion-matrix output from DIRECTION_TUGRUL.md.

Prerequisites:
    python build_dataset_recordings.py   # produces dataset_recordings.npz

Usage (run from tugrul_paket/model_and_shi/):
    python retrain_evaluate.py
    python retrain_evaluate.py --npz dataset_recordings.npz --epochs 60
    python retrain_evaluate.py --skip_cross_session   # within-session only

Outputs (../vad_compare_output/):
    confusion_matrix_within_session.png
    confusion_matrix_cross_session.png   (unless --skip_cross_session)
    Appends real accuracy numbers to vad_compare_report.md
"""

import os
import argparse
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ElectrolarynxCNN
from dataset import ElectrolarynxDataset
from evaluation import get_within_session_splits, get_cross_session_splits

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, "..", "vad_compare_output")

LABEL_NAMES = ["EVET", "HAYIR", "MERHABA", "TESEKKURLER", "SU",
               "YARDIM", "TAMAM", "DUR", "GEL", "GUNAYDIN"]


# ── Training helpers ───────────────────────────────────────────────────────────

def _make_model(device):
    return ElectrolarynxCNN(num_classes=10, in_channels=3).to(device)


def _fit(model, X_tr, y_tr, X_vl, y_vl, epochs, patience, device):
    tr_loader = DataLoader(ElectrolarynxDataset(X_tr, y_tr, is_train=True),
                           batch_size=16, shuffle=True)
    vl_loader = DataLoader(ElectrolarynxDataset(X_vl, y_vl, is_train=False),
                           batch_size=32)
    criterion = nn.CrossEntropyLoss()
    opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss, best_w, no_imp = float('inf'), None, 0

    for ep in range(epochs):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(Xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        vl_loss, total = 0.0, 0
        with torch.no_grad():
            for Xb, yb in vl_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                vl_loss += criterion(model(Xb), yb).item() * len(yb)
                total   += len(yb)
        vl_loss /= total

        if vl_loss < best_loss:
            best_loss, best_w, no_imp = vl_loss, deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    model.load_state_dict(best_w)
    return model


def _predict_probs(model, X, device):
    model.eval()
    loader = DataLoader(ElectrolarynxDataset(X, np.zeros(len(X), np.int64), is_train=False),
                        batch_size=32)
    parts = []
    with torch.no_grad():
        for Xb, _ in loader:
            parts.append(torch.softmax(model(Xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(parts)


def topk(probs, y, k):
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([y[i] in top_k[i] for i in range(len(y))]))


# ── Within-session 5-fold CV ───────────────────────────────────────────────────

def within_session_cv(X, y, sid, epochs, patience, device):
    sessions = np.unique(sid)
    all_true, all_pred = [], []
    ws_top1, ws_top3 = {}, {}

    for s in sessions:
        mask   = sid == s
        Xs, ys = X[mask], y[mask]
        fold_true, fold_pred = [], []

        for fold_i, (tr_idx, te_idx) in enumerate(
                get_within_session_splits(Xs, ys, n_splits=5)):
            print(f"  kayit{s}  fold {fold_i+1}/5", end='\r', flush=True)
            model  = _make_model(device)
            model  = _fit(model, Xs[tr_idx], ys[tr_idx],
                          Xs[te_idx], ys[te_idx], epochs, patience, device)
            probs  = _predict_probs(model, Xs[te_idx], device)
            fold_true.extend(ys[te_idx])
            fold_pred.extend(probs)

        ft = np.array(fold_true)
        fp = np.array(fold_pred)
        all_true.extend(ft)
        all_pred.extend(fp)
        ws_top1[s] = topk(fp, ft, 1)
        ws_top3[s] = topk(fp, ft, 3)
        print(f"  kayit{s}:  Top-1={ws_top1[s]:.1%}  Top-3={ws_top3[s]:.1%}          ")

    return np.array(all_true), np.array(all_pred), ws_top1, ws_top3


# ── Cross-session LOSO ─────────────────────────────────────────────────────────

def cross_session_loso(X, y, sid, epochs, patience, device):
    sessions     = np.unique(sid)
    session_data = {f"kayit{int(s)}": (X[sid == s], y[sid == s]) for s in sessions}
    splits       = get_cross_session_splits(session_data)

    all_true, all_pred = [], []
    cs_top1 = {}

    for split in splits:
        test_name = split['test_session_name']
        X_tr, y_tr = split['train_X'], split['train_y']
        X_te, y_te = split['test_X'],  split['test_y']

        # Hold out 10% of training set for early-stopping val (not test data)
        perm  = np.random.default_rng(42).permutation(len(X_tr))
        n_vl  = max(4, len(X_tr) // 10)
        X_vl, y_vl = X_tr[perm[:n_vl]], y_tr[perm[:n_vl]]
        X_tr2, y_tr2 = X_tr[perm[n_vl:]], y_tr[perm[n_vl:]]

        print(f"  LOSO held-out: {test_name}  "
              f"(train={len(X_tr2)}, val={len(X_vl)}, test={len(X_te)})")
        model = _make_model(device)
        model = _fit(model, X_tr2, y_tr2, X_vl, y_vl, epochs, patience, device)
        probs = _predict_probs(model, X_te, device)
        t1    = topk(probs, y_te, 1)
        cs_top1[test_name] = t1
        all_true.extend(y_te)
        all_pred.extend(probs)
        print(f"    {test_name}: Top-1={t1:.1%}")

    return np.array(all_true), np.array(all_pred), cs_top1


# ── Confusion matrix plot ──────────────────────────────────────────────────────

def plot_confusion_matrix(true_labels, pred_probs, title, path):
    preds = np.argmax(pred_probs, axis=1)
    n  = len(LABEL_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, preds):
        cm[t, p] += 1

    row_sums  = cm.sum(axis=1, keepdims=True).clip(1)
    cm_norm   = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(LABEL_NAMES, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i,
                        f"{cm_norm[i,j]:.0%}\n({cm[i,j]})",
                        ha='center', va='center', fontsize=7,
                        color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ── Update report ──────────────────────────────────────────────────────────────

def _update_report(ws_top1_overall, ws_top3_overall, cs_top1_str):
    report_path = os.path.join(OUTPUT_DIR, 'vad_compare_report.md')
    if not os.path.exists(report_path):
        return
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    new_row = (f"| New VAD segments  | {ws_top1_overall:.1%} "
               f"| {ws_top3_overall:.1%} | {cs_top1_str} |")
    content = content.replace(
        "| New VAD segments  | _run train_base_model.py_ | — | — |",
        new_row)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Report updated: {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',                default='dataset_recordings.npz')
    parser.add_argument('--epochs',             type=int, default=60)
    parser.add_argument('--patience',           type=int, default=15)
    parser.add_argument('--skip_cross_session', action='store_true',
                        help='Skip LOSO cross-session eval (faster)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | epochs={args.epochs} | patience={args.patience}")

    d = np.load(args.npz)
    X, y, sid = d['X'], d['y'], d['session_id']
    print(f"Dataset: {len(X)} segments | sessions {np.unique(sid).tolist()}")

    # ─── A) Within-session 5-fold CV ──────────────────────────────────────────
    print("\n=== A) Within-session 5-fold CV ===")
    ws_true, ws_pred, ws_top1, ws_top3 = within_session_cv(
        X, y, sid, args.epochs, args.patience, device)

    ws_top1_all = topk(ws_pred, ws_true, 1)
    ws_top3_all = topk(ws_pred, ws_true, 3)
    print(f"\nWithin-session overall:  Top-1={ws_top1_all:.1%}  Top-3={ws_top3_all:.1%}")
    print(f"Baseline (prev):         Top-1=67%  Top-3=87%")

    plot_confusion_matrix(
        ws_true, ws_pred,
        f"Within-session confusion (new VAD segs)  Top-1={ws_top1_all:.1%}",
        os.path.join(OUTPUT_DIR, 'confusion_matrix_within_session.png'),
    )

    # ─── B) Cross-session LOSO ────────────────────────────────────────────────
    cs_top1_str = "—"
    if not args.skip_cross_session:
        print("\n=== B) Cross-session LOSO ===")
        cs_true, cs_pred, cs_top1 = cross_session_loso(
            X, y, sid, args.epochs, args.patience, device)
        cs_top1_all = topk(cs_pred, cs_true, 1)
        cs_top1_str = f"{cs_top1_all:.1%}"
        print(f"\nCross-session overall:  Top-1={cs_top1_all:.1%}")
        print(f"Baseline (prev):        Top-1=52%")

        plot_confusion_matrix(
            cs_true, cs_pred,
            f"Cross-session LOSO (new VAD segs)  Top-1={cs_top1_all:.1%}",
            os.path.join(OUTPUT_DIR, 'confusion_matrix_cross_session.png'),
        )

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n=== ACCURACY COMPARISON ===")
    print(f"{'Setup':<32} {'Top-1':>7} {'Top-3':>7} {'X-sess Top-1':>14}")
    print(f"{'Previous baseline':<32} {'67%':>7} {'87%':>7} {'52%':>14}")
    print(f"{'New VAD segments':<32} {ws_top1_all:>7.1%} {ws_top3_all:>7.1%} {cs_top1_str:>14}")

    _update_report(ws_top1_all, ws_top3_all, cs_top1_str)


if __name__ == '__main__':
    main()
