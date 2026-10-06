import pandas as pd
import numpy as np
import os
from evaluation import get_within_session_splits
from rf_baseline import train_evaluate_rf

# 1. Kelime Metni -> Sayısal ID Eşleştirmesi (Mapping)
LEGACY_LABEL_MAP = {
    "EVET": 0, "HAYIR": 1, "MERHABA": 2, "TESEKKURLER": 3, "SU": 4,
    "YARDIM": 5, "TAMAM": 6, "DUR": 7, "GEL": 8, "GUNAYDIN": 9
}


def load_legacy_data(session_dir):
    """
    Eski 3 kanallı veriyi yükler ve metin etiketlerini sayıya çevirir.
    """
    raw_path = os.path.join(session_dir, "raw_legacy.csv")
    markers_path = os.path.join(session_dir, "markers.csv")

    if not os.path.exists(raw_path) or not os.path.exists(markers_path):
        print("HATA: Dosyalar bulunamadı! Lütfen recordings/legacy_session klasörünü kontrol et.")
        return None, None

    # --- 1. Ham Veriyi Oku ---
    df_raw = pd.read_csv(raw_path)
    df_raw.columns = df_raw.columns.str.strip()
    legacy_channels = ["pzt1", "pzt2", "emg1"]
    raw_values = df_raw[legacy_channels].values

    # --- 2. Etiketleri Oku ve Temizle ---
    df_markers = pd.read_csv(markers_path)
    df_markers.columns = df_markers.columns.str.strip()

    clean_labels = []
    for raw_label in df_markers['label']:
        # 'EVET_REP1' gibi metinleri parçala, sadece 'EVET' kısmını al
        # Not: SESSIZLIK_BASLANGIC gibi kelime olmayan satırları atla
        if "REP" in raw_label:
            word_part = raw_label.split('_')[0]  # 'EVET' kısmını çeker
            label_id = LEGACY_LABEL_MAP.get(word_part, -1)
            if label_id != -1:
                clean_labels.append(label_id)
        else:
            # Eğer format direkt 'EVET' ise:
            label_id = LEGACY_LABEL_MAP.get(raw_label, -1)
            if label_id != -1:
                clean_labels.append(label_id)

    # --- 3. Segmentlere Böl ---
    num_segments = len(clean_labels)
    # Eski veride her kelimenin 1000 sample olduğunu varsayıyoruz (2 saniye)
    X_3ch = raw_values[:num_segments * 1000].reshape(num_segments, 1000, 3)
    X_3ch = X_3ch.transpose(0, 2, 1)  # (N, 3, 1000)

    # --- 4. 6 Kanala Tamamla (Zero-Padding) ---
    X_6ch = np.zeros((num_segments, 6, 1000))
    X_6ch[:, :3, :] = X_3ch

    y = np.array(clean_labels)
    return X_6ch, y


def run_test():
    session_dir = "recordings/legacy_session"
    X, y = load_legacy_data(session_dir)

    if X is None or len(X) == 0:
        print("HATA: İşlenecek veri bulunamadı.")
        return

    print(f"Legacy Veri Yüklendi. Segment: {len(X)} | Kanal: 3 (6'ya tamamlandı)")
    print("Within-session (5-Fold CV) Başlıyor...\n")

    splits = get_within_session_splits(X, y, n_splits=5)

    accuracies = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"--- FOLD {fold + 1} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        _, acc = train_evaluate_rf(X_train, y_train, X_test, y_test)
        accuracies.append(acc)

    print("\n" + "=" * 40)
    print(f"LEGACY WITHIN-SESSION SONUCU: %{np.mean(accuracies) * 100:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    run_test()