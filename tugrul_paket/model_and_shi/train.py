import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


def train_model(model, train_loader, val_loader, num_epochs=100, patience=15, device='cpu', use_wandb=False):
    """
    Modeli eğitir, en iyi epoch'u bulur ve o ağırlıkları döndürür.
    """
    # Kayıp fonksiyonu ve Optimizasyon
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.to(device)

    best_val_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0

    print(f"Eğitim başlıyor... Cihaz: {device} | Max Epoch: {num_epochs}")

    for epoch in range(num_epochs):
        # ==================== TRAIN (EĞİTİM) AŞAMASI ====================
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == y_batch.data).item()
            total_train += y_batch.size(0)

        train_loss = train_loss / total_train
        train_acc = correct_train / total_train

        # ==================== VALIDATION (DOĞRULAMA) AŞAMASI ====================
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == y_batch.data).item()
                total_val += y_batch.size(0)

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        # Öğrenme oranını güncelle
        scheduler.step()

        print(f"Epoch {epoch + 1:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # WandB loglama (opsiyonel)
        if use_wandb:
            import wandb
            wandb.log({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            })

        # ==================== EARLY STOPPING KONTROLÜ ====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # En iyi modelin kopyasını al
            best_model_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\nEarly Stopping tetiklendi! {patience} epoch boyunca iyileşme olmadı. (Durdurulan Epoch: {epoch + 1})")
                break

    print(f"\nEğitim tamamlandı. En iyi Val Loss: {best_val_loss:.4f}")
    # Eğitimi bitirirken, aşırı öğrenmemiş olan "en iyi" ağırlıkları modele geri yükle
    model.load_state_dict(best_model_weights)

    return model