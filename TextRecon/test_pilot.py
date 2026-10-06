import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from signal_processing import preprocess
from vad import VAD, VAD_CONFIG


def test_pilot_data(session_dir="recordings/pilot_session"):
    # Dosya adının yeni formata göre raw_5ch.csv olacağı varsayıldı
    raw_csv_path = os.path.join(session_dir, "raw_5ch.csv")

    if not os.path.exists(raw_csv_path):
        print(f"HATA: {raw_csv_path} bulunamadı! Klasör ve dosya adını doğru yazdığından emin ol.")
        # Eğer Serhat ismini değiştirmeyi unutup raw_6ch.csv atarsa diye alternatif yol:
        alt_path = os.path.join(session_dir, "raw_6ch.csv")
        if os.path.exists(alt_path):
            print("UYARI: raw_5ch.csv bulunamadı ama raw_6ch.csv bulundu, o okunuyor...")
            raw_csv_path = alt_path
        else:
            return

    print("1. Veri yükleniyor...")
    df = pd.read_csv(raw_csv_path, sep=',')
    df.columns = df.columns.str.strip()

    # emg2 ÇIKARILDI! Artık sadece 5 kanal okuyoruz.
    hedef_kolonlar = ["emg1", "pzt1", "pzt2", "pzt3", "pzt4"]
    raw_data = df[hedef_kolonlar].values.astype(np.int16)

    print("EMG1 İlk 20 değer (Ham ADC):", raw_data[:20, 0])

    total_samples = raw_data.shape[0]
    fs = VAD_CONFIG["fs"]
    print(f"Toplam süre: {total_samples / fs:.2f} saniye ({total_samples} sample)")

    print("2. Sinyal filtreleniyor (Band-pass + Notch)...")
    processed_signal = preprocess(raw_data, fs=fs)

    print("3. VAD Kalibrasyonu (İlk 20 saniye mutlak sessizlik)...")
    calibration_samples = 20 * fs
    if total_samples < calibration_samples:
        print("HATA: Veri 20 saniyeden kısa! Serhat protokolü yanlış uygulamış.")
        return

    vad = VAD(VAD_CONFIG)
    vad.calibrate(processed_signal[:calibration_samples])

    print("4. Kelime tespiti yapılıyor...")
    test_signal = processed_signal[calibration_samples:]
    segments = vad.process_offline(test_signal)

    print(f"\n--- TEST SONUCU ---")
    print(f"Bulunan toplam kelime segmenti sayısı: {len(segments)}")

    if len(segments) > 0:
        avg_dur = np.mean([(seg.end_idx - seg.start_idx) for seg in segments])
        print(f"Ortalama kelime süresi: {avg_dur / fs:.2f} saniye")

    # 5. Görselleştirme
    print("\nGrafik çiziliyor, lütfen bekleyin...")
    plt.figure(figsize=(15, 5))

    time_axis = np.arange(len(test_signal)) / fs

    # DÜZELTME: Artık 2. kanal ([:, 1]) PZT olduğu için onu EMG ile toplamıyoruz.
    # Sadece 0. kanal (EMG1) üzerinden mutlak değer alıyoruz.
    emg_abs = np.abs(test_signal[:, 0])
    plt.plot(time_axis, emg_abs, label="EMG1 (Mutlak)", color='lightgray')

    plt.axhline(y=vad.threshold, color='r', linestyle='--', label=f"Threshold ({vad.threshold:.4f})")

    for i, seg in enumerate(segments):
        start_sec = (seg.start_idx - calibration_samples) / fs
        end_sec = (seg.end_idx - calibration_samples) / fs
        label = "Bulunan Kelimeler" if i == 0 else ""
        plt.axvspan(start_sec, end_sec, color='green', alpha=0.3, label=label)

    plt.title("Pilot Veri VAD Testi (20. saniyeden sonrası) - 5 Kanal")
    plt.xlabel("Zaman (Saniye)")
    plt.ylabel("Genlik")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_pilot_data()