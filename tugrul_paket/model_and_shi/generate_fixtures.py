import numpy as np
import json
import os
import pandas as pd
from signal_processing import preprocess
from vad import VAD, VAD_CONFIG


def generate_synthetic_data(fs=500):
    """
    Test için sahte 6 kanallı ham ADC verisi üretir.
    Senaryo: 20 saniye baseline, ardından 3 farklı uzunlukta kelime aktivasyonu.
    """
    total_samples = 30 * fs  # Toplam 30 saniyelik kayıt
    data = np.random.normal(2048, 10, (total_samples, 6))  # ADC orta noktası (2048) etrafında gürültü

    # Kelime 1 (Kısa Kelime - Örn: "SU") -> 22. saniyede, 400ms sürsün
    start1, end1 = 22 * fs, int(22.4 * fs)
    data[start1:end1, :2] += np.random.normal(500, 100, (end1 - start1, 2))  # EMG kanallarına güç bindir

    # Kelime 2 (Normal Kelime - Örn: "MERHABA") -> 24. saniyede, 800ms sürsün
    start2, end2 = 24 * fs, int(24.8 * fs)
    data[start2:end2, :2] += np.random.normal(800, 150, (end2 - start2, 2))

    # Kelime 3 (Çok Uzun Kelime/Gürültü) -> 27. saniyede, 2.5 saniye sürsün (MAX_DURATION'da kesilmeli)
    start3, end3 = 27 * fs, int(29.5 * fs)
    data[start3:end3, :2] += np.random.normal(600, 100, (end3 - start3, 2))

    # 12-bit sınırlarına kırp ve int16 yap
    data = np.clip(data, 0, 4095).astype(np.int16)
    return data


def create_fixtures(output_dir="vad_test_fixtures"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sahte Veriyi Üret
    raw_data = generate_synthetic_data()

    # 2. Ham Veriyi ve Baseline'ı Kaydet
    df_raw = pd.DataFrame(raw_data, columns=["emg1", "emg2", "pzt1", "pzt2", "pzt3", "pzt4"])
    df_raw.to_csv(os.path.join(output_dir, "fixture_01_raw.csv"), index=False)

    # Baseline ilk 20 saniye (10000 sample)
    df_baseline = df_raw.iloc[:10000]
    df_baseline.to_csv(os.path.join(output_dir, "fixture_01_baseline.csv"), index=False)

    # 3. Python Pipeline'ından Geçir (Doğruluk Referansı)
    processed_signal = preprocess(raw_data)
    vad = VAD(VAD_CONFIG)
    vad.calibrate(processed_signal[:10000])
    segments = vad.process_offline(processed_signal[10000:])

    # 4. Beklenen Segment Çıktılarını JSON Olarak Kaydet
    # Çıkan indeksleri (10000 eklendiğine dikkat, çünkü offset var)
    expected_segments = []
    for seg in segments:
        expected_segments.append({
            "start_idx": seg.start_idx + 10000,
            "end_idx": seg.end_idx + 10000,
            "duration_samples": seg.end_idx - seg.start_idx
        })

    with open(os.path.join(output_dir, "fixture_01_expected.json"), "w") as f:
        json.dump({
            "threshold": float(vad.threshold),
            "segments": expected_segments
        }, f, indent=4)

    print(f"Fixture'lar '{output_dir}' klasörüne başarıyla üretildi.")
    print(f"Test Referansı Olarak Bulunan Segmentler: {expected_segments}")


if __name__ == "__main__":
    create_fixtures()