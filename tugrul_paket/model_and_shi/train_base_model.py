"""s3 hariç 3 oturum ile base model eğitir, ağırlıkları kaydeder.
Demo'dan önce bu base model fine-tune edilecek."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ElectrolarynxCNN
from dataset import ElectrolarynxDataset


def main(npz="dataset_poc.npz", out="base_model.pth", epochs=60,
         exclude_sessions=(3,)):
    d = np.load(npz)
    X, y, sid = d['X'], d['y'], d['session_id']

    # Outlier oturumları çıkar
    if exclude_sessions:
        mask = ~np.isin(sid, exclude_sessions)
        X, y = X[mask], y[mask]
        print(f"Excluded sessions {exclude_sessions}: {len(X)} segments remaining")

    print(f"Base model training: {len(X)} segments, {epochs} epochs")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    ds = ElectrolarynxDataset(X, y, is_train=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = ElectrolarynxCNN(num_classes=10, in_channels=3).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    model.train()
    for ep in range(epochs):
        total_loss = correct = total = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(Xb)                # 'out' yerine 'preds' yaptık
            loss = crit(preds, yb)           # 'out' yerine 'preds' yaptık
            loss.backward()
            opt.step()
            total_loss += loss.item() * Xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item() # 'out' yerine 'preds' yaptık
            total += yb.size(0)
        sched.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={total_loss/total:.4f} train_acc={correct/total*100:.1f}%")

    torch.save(model.state_dict(), out)
    print(f"\nBase model kaydedildi: {out}")
    print("Demo öncesi kullanım:")
    print("  python finetune_calibration.py --sensors KALIBRASYON_SENSORS.csv --markers KALIBRASYON_MARKERS.csv")


if __name__ == "__main__":
    main()
