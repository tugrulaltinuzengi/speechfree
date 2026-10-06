import pandas as pd
import numpy as np
from evaluation import get_within_session_splits
from rf_baseline import train_evaluate_rf


def load_old_csv_data(file_path):
    """
    Eski 3 kanallı CSV formatını okur ve (N, 3, 1000) formatına çevirir.
    Dosyada timestamp_s, pzt1, pzt2, emg1 sütunları olduğunu varsayıyoruz.
    """
    print(f"Eski veri yükleniyor: {file_path}")
    df = pd.read_csv(file_path, sep=',')
    df.columns = df.columns.str.strip()

    # Sadece ihtiyacımız olan 3 kanalı seçiyoruz (Zaman damgasını atıyoruz)
    # Sıralama: pzt1, pzt2, emg1
    channels = ["pzt1", "pzt2", "emg1"]
    raw_values = df[channels].values

    # EĞER bu dosya tek bir uzun kayıt ise (segmentlere ayrılmamışsa):
    # Bu basit test için veriyi 1000'er örneklem lik parçalara bölelim.
    num_samples = len(raw_values)
    num_segments = num_samples // 1000

    X = raw_values[:num_segments * 1000].reshape(num_segments, 1000, 3)
    # Model (Channels, Samples) beklediği için transpose yapıyoruz: (N, 3, 1000)
    X = X.transpose(0, 2, 1)

    # Eski veride etiketler yoksa, test amaçlı rastgele etiketler atayalım
    # (Gerçek sonuç için markers.csv gibi bir dosyan olmalı)
    y = np.random.randint(0, 10, size=num_segments)

    return X, y


def run_legacy_test():
    # Buraya kendi dosya yolunu yaz
    X, y = load_old_csv_data("eski_verin.csv")

    print(f"Veri hazır! Segment sayısı: {len(X)}, Kanal sayısı: {X.shape[1]}")

    # KRİTİK NOT: rf_baseline.py kodumuz 6 kanal bekliyor.
    # Eski veride 3 kanal olduğu için, rf_baseline.py içindeki
    # 'extract_features_from_segment' fonksiyonunu 3 kanala göre modifiye etmemiz gerekebilir
    # veya veriyi 6 kanala 'sıfır' ekleyerek tamamlayabiliriz (Padding).

    # 3 kanalı 6 kanala tamamlayalım (Geri kalan 3 kanala sessizlik/sıfır ekle)
    X_padded = np.zeros((X.shape[0], 6, 1000))
    X_padded[:, :3, :] = X  # İlk 3 kanalı doldur

    print("3 kanallı veri 6 kanala uyarlandı (Zero-padding).")

    splits = get_within_session_splits(X_padded, y, n_splits=5)

    fold_accuracies = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n--- FOLD {fold + 1} ---")
        X_train, y_train = X_padded[train_idx], y[train_idx]
        X_test, y_test = X_padded[test_idx], y[test_idx]

        _, acc = train_evaluate_rf(X_train, y_train, X_test, y_test)
        fold_accuracies.append(acc)

    print(f"\nOrtalama Başarı: %{np.mean(fold_accuracies) * 100:.2f}")


if __name__ == "__main__":
    run_legacy_test()