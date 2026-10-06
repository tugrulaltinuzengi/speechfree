import os
import glob
import pandas as pd
import numpy as np

from signal_processing import preprocess
from vad import VAD, VAD_CONFIG


def build_full_dataset(base_dir="recordings", output_file="dataset.npz"):
    """
    Tüm session klasörlerini gezer, VAD ile kelimeleri keser,
    Serhat'ın yeni markers.csv formatıyla eşleştirir ve kaydeder.
    """
    if not os.path.exists(base_dir):
        print(f"HATA: '{base_dir}' klasörü bulunamadı!")
        return

    # Kelime -> ID Eşleştirmesi (Mapping)
    LABEL_MAPPING = {
        "EVET": 0, "HAYIR": 1, "MERHABA": 2, "TESEKKURLER": 3, "SU": 4,
        "YARDIM": 5, "TAMAM": 6, "DUR": 7, "GEL": 8, "GUNAYDIN": 9
    }

    session_folders = sorted(glob.glob(os.path.join(base_dir, "session_*")))

    if len(session_folders) == 0:
        print(f"HATA: '{base_dir}' içinde 'session_' ile başlayan klasör bulunamadı.")
        return

    print(f"Toplam {len(session_folders)} oturum bulundu. İşlem başlatılıyor...\n")

    X_all = []
    y_all = []

    fs = VAD_CONFIG["fs"]
    calibration_samples = 20 * fs

    for session_path in session_folders:
        session_name = os.path.basename(session_path)
        raw_csv_path = os.path.join(session_path, "raw_3ch.csv")
        markers_csv_path = os.path.join(session_path, "markers.csv")

        if not os.path.exists(raw_csv_path) or not os.path.exists(markers_csv_path):
            print(f"[ATLANDI] {session_name}: Dosyalar eksik.")
            continue

        print(f"--- İşleniyor: {session_name} ---")

        # 1. Veriyi Yükle ve Filtrele
        df_raw = pd.read_csv(raw_csv_path, sep=',')
        df_raw.columns = df_raw.columns.str.strip()

        # 3 Kanal yapısı: emg1, pzt1, pzt2 (preprocess() bu sıralamayı bekliyor)
        hedef_kolonlar = ["emg1", "pzt1", "pzt2"]

        try:
            raw_data = df_raw[hedef_kolonlar].values.astype(np.int16)
        except KeyError as e:
            print(f"[HATA] {session_name}: Sütun ismi bulunamadı ({e})! Atlanıyor.")
            continue

        processed_signal = preprocess(raw_data, fs=fs)

        # 2. VAD İşlemleri
        if len(processed_signal) < calibration_samples:
            print(f"[HATA] {session_name}: Veri çok kısa. Atlanıyor.")
            continue

        vad = VAD(VAD_CONFIG)
        vad.calibrate(processed_signal[:calibration_samples])

        test_signal = processed_signal[calibration_samples:]
        segments = vad.process_offline(test_signal)

        # 3. Etiketleri Oku (Yeni Serhat Formatı)
        df_markers = pd.read_csv(markers_csv_path, sep=',')
        df_markers.columns = df_markers.columns.str.strip()

        # Sadece 'window_start' olan satırları filtreliyoruz (Çift sayımı engellemek için)
        word_starts = df_markers[df_markers['event'] == 'window_start']

        expected_labels = []
        for word_text in word_starts['word']:
            if pd.isna(word_text): continue

            word_clean = str(word_text).strip().upper()
            label_id = LABEL_MAPPING.get(word_clean, -1)

            if label_id != -1:
                expected_labels.append(label_id)

        # 4. Eşleştirme ve Kayıt
        if len(segments) != len(expected_labels):
            print(
                f"[UYARI] {session_name}: VAD {len(segments)} segment buldu, markers.csv'de {len(expected_labels)} etiket var!")

        min_len = min(len(segments), len(expected_labels))
        for i in range(min_len):
            seg = segments[i]
            label = expected_labels[i]

            # seg.data shape: (1000, 3) -> (3, 1000) (Conv1d formatı)
            tensor_data = seg.data.T
            X_all.append(tensor_data)
            y_all.append(label)

        print(f"[BAŞARILI] {session_name}: {min_len} segment eklendi.\n")

    if len(X_all) > 0:
        X_array = np.array(X_all, dtype=np.float32)
        y_array = np.array(y_all, dtype=np.int64)

        print("=========================================")
        print(f"VERİ DERLEME TAMAMLANDI!")
        print(f"X Boyutu: {X_array.shape} (Örnek, Kanal, Sample)")
        print(f"y Boyutu: {y_array.shape}")

        np.savez(output_file, X=X_array, y=y_array)
        print(f"'{output_file}' dosyası güncellendi. Artık testi başlatabilirsin!")
    else:
        print("HATA: Hiçbir geçerli segment derlenemedi.")


if __name__ == "__main__":
    build_full_dataset()