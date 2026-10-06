"""PoC dataset ile within-session 5-fold ve cross-session LOSO eğitimi."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import ElectrolarynxCNN
from dataset import ElectrolarynxDataset
from train import train_model
from evaluation import get_within_session_splits, get_cross_session_splits


def topk_accuracy(model, loader, device, k=3):
    model.eval()
    correct1 = correctk = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            top = logits.topk(k, dim=1).indices
            correct1 += (top[:, 0] == y).sum().item()
            correctk += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
    return correct1 / total, correctk / total


def run_within_session(npz="dataset_poc.npz", epochs=80):
    d = np.load(npz); X, y = d['X'], d['y']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== WITHIN-SESSION 5-FOLD CNN === device={device}")

    accs, t3s = [], []
    for f, (tr, te) in enumerate(get_within_session_splits(X, y, 5)):
        tr_loader = DataLoader(ElectrolarynxDataset(X[tr], y[tr], is_train=True),
                               batch_size=16, shuffle=True)
        te_loader = DataLoader(ElectrolarynxDataset(X[te], y[te], is_train=False),
                               batch_size=16, shuffle=False)
        model = ElectrolarynxCNN(num_classes=10, in_channels=3)
        model = train_model(model, tr_loader, te_loader,
                            num_epochs=epochs, patience=15, device=device)
        a, t3 = topk_accuracy(model, te_loader, device, k=3)
        print(f"FOLD {f+1}: Top1={a*100:.2f}%  Top3={t3*100:.2f}%\n")
        accs.append(a); t3s.append(t3)
    print(f"\nCNN WITHIN-SESSION: Top1={np.mean(accs)*100:.2f}%  Top3={np.mean(t3s)*100:.2f}%")
    return np.mean(accs), np.mean(t3s)


def run_cross_session(npz="dataset_poc.npz", epochs=80):
    d = np.load(npz); X, y, sid = d['X'], d['y'], d['session_id']
    sessions = {f"s{s}": (X[sid == s], y[sid == s]) for s in np.unique(sid)}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== CROSS-SESSION LOSO CNN === {len(sessions)} sessions")

    accs, t3s = [], []
    for sp in get_cross_session_splits(sessions):
        tr_loader = DataLoader(ElectrolarynxDataset(sp['train_X'], sp['train_y'], is_train=True),
                               batch_size=16, shuffle=True)
        te_loader = DataLoader(ElectrolarynxDataset(sp['test_X'], sp['test_y'], is_train=False),
                               batch_size=16, shuffle=False)
        model = ElectrolarynxCNN(num_classes=10, in_channels=3)
        model = train_model(model, tr_loader, te_loader,
                            num_epochs=epochs, patience=15, device=device)
        a, t3 = topk_accuracy(model, te_loader, device, k=3)
        print(f"Test={sp['test_session_name']}: Top1={a*100:.2f}%  Top3={t3*100:.2f}%\n")
        accs.append(a); t3s.append(t3)
    print(f"\nCNN CROSS-SESSION: Top1={np.mean(accs)*100:.2f}%  Top3={np.mean(t3s)*100:.2f}%")
    return np.mean(accs), np.mean(t3s)


if __name__ == "__main__":
    run_within_session()
    run_cross_session()
