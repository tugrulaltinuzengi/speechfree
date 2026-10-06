import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Sklearn ve Numpy'dan gelen gereksiz matematiksel uyarıları temizle
warnings.filterwarnings("ignore")

# Eğer evaluation.py içinden import ediyorsan:
from evaluation import get_within_session_splits


def extract_features_from_segment(segment_5ch, fs=500):
    """
    Giriş: (5, 1000) boyutunda numpy array
    Çıkış: ~61 uzunluğunda 1D feature vektörü
    """
    features = []

    # 5 Kanal üzerinden özellikleri çıkar
    for i in range(5):
        ch_data = segment_5ch[i, :]

        # 1. Zaman Uzayı
        rms = np.sqrt(np.mean(ch_data ** 2))
        mav = np.mean(np.abs(ch_data))
        zcr = np.sum(np.diff(np.sign(ch_data)) != 0) / len(ch_data)
        wl = np.sum(np.abs(np.diff(ch_data)))

        # 2. Hjorth Parametreleri
        activity = np.var(ch_data)
        dx = np.diff(ch_data)
        mobility = np.std(dx) / np.std(ch_data) if np.std(ch_data) > 0 else 0
        ddx = np.diff(dx)
        complexity = (np.std(ddx) / np.std(dx)) / mobility if mobility > 0 and np.std(dx) > 0 else 0

        # 3. Frekans Uzayı (Band Power)
        freqs, psd = signal.welch(ch_data, fs, nperseg=256)
        mean_freq = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0

        bp1 = np.sum(psd[(freqs >= 5) & (freqs < 20)])
        bp2 = np.sum(psd[(freqs >= 20) & (freqs < 50)])
        bp3 = np.sum(psd[(freqs >= 50) & (freqs < 100)])
        bp4 = np.sum(psd[(freqs >= 100) & (freqs <= 200)])

        features.extend([rms, mav, zcr, wl, activity, mobility, complexity, mean_freq, bp1, bp2, bp3, bp4])

    # 4. Cross-Channel (Kanal arası ilişkiler)
    # Yeni 5 kanallı sistemde: EMG1(0), PZT1(1), PZT2(2), PZT3(3), PZT4(4)
    # PZT1 ve PZT4 arası korelasyon
    pzt_corr = np.corrcoef(segment_5ch[1, :], segment_5ch[4, :])[0, 1]
    pzt_corr = np.nan_to_num(pzt_corr)

    features.append(pzt_corr)

    return np.array(features)


def train_evaluate_rf(X_train_raw, y_train, X_test_raw, y_test):
    # Verileri 1D Feature vektörlerine dönüştür
    X_train_features = np.array([extract_features_from_segment(seg) for seg in X_train_raw])
    X_test_features = np.array([extract_features_from_segment(seg) for seg in X_test_raw])

    # Random Forest Modelini Eğit
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_features, y_train)

    # Test Et
    y_pred = rf_model.predict(X_test_features)
    acc = accuracy_score(y_test, y_pred)

    return rf_model, acc


def run_new_baseline():
    print("dataset.npz yükleniyor...")
    try:
        data = np.load("dataset.npz")
        X = data['X']
        y = data['y']
    except FileNotFoundError:
        print("HATA: dataset.npz bulunamadı. Lütfen önce build_dataset.py kodunu çalıştır.")
        return

    print(f"Veri Yüklendi! Toplam Segment: {len(X)} | Kanal: {X.shape[1]}")
    print("5-Fold Cross Validation (Within-Session) Testi Başlıyor...\n")

    splits = get_within_session_splits(X, y, n_splits=5)

    accuracies = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"--- FOLD {fold + 1} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        _, acc = train_evaluate_rf(X_train, y_train, X_test, y_test)
        print(f"Fold {fold + 1} Başarısı: %{acc * 100:.2f}")
        accuracies.append(acc)

    print("\n" + "=" * 40)
    print(f"YENİ 5 KANALLI VERİ BASELINE SONUCU: %{np.mean(accuracies) * 100:.2f}")
    print("=" * 40)


# Bu blok, dosya doğrudan çalıştırıldığında testin başlamasını sağlar
if __name__ == "__main__":
    run_new_baseline()