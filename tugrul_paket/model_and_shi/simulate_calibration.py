"""
Demo senaryosunu simüle eder:
1. Bir oturumu "demo günü" olarak seç
2. O oturumun yarısı kalibrasyon, yarısı demo test
3. Diğer oturumlarla base model eğit
4. Kalibrasyon ile fine-tune
5. Demo test'te ölç

Bunu iki configde çalıştır:
  - 4 oturumlu base
  - s3 hariç 3 oturumlu base
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy

from models import ElectrolarynxCNN
from dataset import ElectrolarynxDataset


def split_session_calib_test(X_sess, y_sess, n_calib_per_class=5, seed=42):
    """Bir oturumun verisini per-class olarak kalibrasyon/test böler."""
    rng = np.random.RandomState(seed)
    calib_idx, test_idx = [], []
    for cls in np.unique(y_sess):
        cls_idx = np.where(y_sess == cls)[0]
        rng.shuffle(cls_idx)
        calib_idx.extend(cls_idx[:n_calib_per_class])
        test_idx.extend(cls_idx[n_calib_per_class:])
    return np.array(calib_idx), np.array(test_idx)


def train_base(X, y, epochs=60, device='cpu'):
    """Base model'i tüm verisi ile eğit (val set yok, fixed epoch)."""
    ds = ElectrolarynxDataset(X, y, is_train=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = ElectrolarynxCNN(num_classes=10, in_channels=3).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    model.train()
    for ep in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    return model


def finetune(model, X, y, epochs=30, lr=5e-4, device='cpu'):
    """Base model'i kalibrasyon verisi ile fine-tune et."""
    ds = ElectrolarynxDataset(X, y, is_train=True)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
    return model


def evaluate(model, X, y, device='cpu'):
    model.eval()
    ds = ElectrolarynxDataset(X, y, is_train=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    c1 = c3 = total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            top3 = logits.topk(3, dim=1).indices
            c1 += (top3[:, 0] == yb).sum().item()
            c3 += (top3 == yb.unsqueeze(1)).any(dim=1).sum().item()
            total += yb.size(0)
    return c1 / total, c3 / total


def run_simulation(npz="dataset_poc.npz", base_epochs=60, ft_epochs=30,
                   n_calib_per_class=5):
    d = np.load(npz)
    X, y, sid = d['X'], d['y'], d['session_id']
    sessions = np.unique(sid)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== KALİBRASYON SİMÜLASYONU ===")
    print(f"Cihaz: {device}")
    print(f"Kalibrasyon: {n_calib_per_class} tekrar/sınıf = {n_calib_per_class*10} segment")
    print(f"Base epochs: {base_epochs}, Fine-tune epochs: {ft_epochs}\n")

    # İki config
    configs = {
        "A_4_session_base": list(sessions),                    # tümü
        "B_3_session_no_s3": [s for s in sessions if s != 3],  # s3 yok
    }

    # Demo gününde olası oturumları test et (s3 demo değil, outlier)
    demo_candidates = [s for s in sessions if s != 3]

    results = {}
    for cfg_name, base_sessions in configs.items():
        print(f"\n{'='*60}")
        print(f"CONFIG: {cfg_name}")
        print(f"Base oturumları: {base_sessions}")
        print(f"{'='*60}")

        cfg_results = []
        for demo_sid in demo_candidates:
            if demo_sid not in base_sessions:
                # Demo oturumu base'de değilse atla (B config'de s3 atlanmış olur)
                continue
            print(f"\n--- Demo günü oturumu: s{demo_sid} ---")

            # Demo oturumunun verisi
            demo_mask = sid == demo_sid
            X_demo, y_demo = X[demo_mask], y[demo_mask]

            # Demo oturumunu kalibrasyon + test'e böl
            calib_idx, test_idx = split_session_calib_test(
                X_demo, y_demo, n_calib_per_class=n_calib_per_class, seed=42
            )
            X_calib, y_calib = X_demo[calib_idx], y_demo[calib_idx]
            X_test, y_test = X_demo[test_idx], y_demo[test_idx]
            print(f"  Kalibrasyon: {len(X_calib)} segment, Test: {len(X_test)} segment")

            # Base train set: demo oturumu hariç, base config'deki diğer oturumlar
            base_mask = np.isin(sid, [s for s in base_sessions if s != demo_sid])
            X_base, y_base = X[base_mask], y[base_mask]
            print(f"  Base train: {len(X_base)} segment ({sum(base_mask)} from {[s for s in base_sessions if s != demo_sid]})")

            # 1. Base model eğit
            base_model = train_base(X_base, y_base, epochs=base_epochs, device=device)

            # 2. ZERO-SHOT (kalibrasyon olmadan demo test)
            zs1, zs3 = evaluate(base_model, X_test, y_test, device=device)
            print(f"  Zero-shot:    Top1={zs1*100:.1f}%  Top3={zs3*100:.1f}%")

            # 3. Fine-tune ve tekrar değerlendir
            ft_model = deepcopy(base_model)
            ft_model = finetune(ft_model, X_calib, y_calib,
                                epochs=ft_epochs, lr=5e-4, device=device)
            ft1, ft3 = evaluate(ft_model, X_test, y_test, device=device)
            print(f"  Fine-tuned:   Top1={ft1*100:.1f}%  Top3={ft3*100:.1f}%  (kazanım: +{(ft1-zs1)*100:.1f} puan)")

            cfg_results.append({
                'demo_sid': demo_sid,
                'zero_shot_top1': zs1, 'zero_shot_top3': zs3,
                'finetuned_top1': ft1, 'finetuned_top3': ft3,
            })

        # Config özeti
        avg_zs1 = np.mean([r['zero_shot_top1'] for r in cfg_results])
        avg_zs3 = np.mean([r['zero_shot_top3'] for r in cfg_results])
        avg_ft1 = np.mean([r['finetuned_top1'] for r in cfg_results])
        avg_ft3 = np.mean([r['finetuned_top3'] for r in cfg_results])
        print(f"\n  >>> {cfg_name} ORTALAMA")
        print(f"      Zero-shot:  Top1={avg_zs1*100:.2f}%  Top3={avg_zs3*100:.2f}%")
        print(f"      Fine-tuned: Top1={avg_ft1*100:.2f}%  Top3={avg_ft3*100:.2f}%")
        results[cfg_name] = cfg_results

    # Final tablo
    print(f"\n\n{'='*60}")
    print("FİNAL KARŞILAŞTIRMA")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'ZS Top1':<10} {'ZS Top3':<10} {'FT Top1':<10} {'FT Top3':<10}")
    for cfg_name, rs in results.items():
        zs1 = np.mean([r['zero_shot_top1'] for r in rs]) * 100
        zs3 = np.mean([r['zero_shot_top3'] for r in rs]) * 100
        ft1 = np.mean([r['finetuned_top1'] for r in rs]) * 100
        ft3 = np.mean([r['finetuned_top3'] for r in rs]) * 100
        print(f"{cfg_name:<25} {zs1:<10.2f} {zs3:<10.2f} {ft1:<10.2f} {ft3:<10.2f}")
    print(f"{'='*60}")
    print("\nZS = Zero-shot (kalibrasyon yok, soğuk start)")
    print("FT = Fine-tuned (kalibrasyon ile fine-tune sonrası)")


if __name__ == "__main__":
    run_simulation()
