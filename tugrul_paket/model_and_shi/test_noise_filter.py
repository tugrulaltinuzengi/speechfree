import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
from signal_processing import preprocess


def analyze_mains_noise(session_dir="recordings/pilot_session"):
    raw_csv_path = os.path.join(session_dir, "raw_3ch.csv")

    if not os.path.exists(raw_csv_path):
        print(f"HATA: {raw_csv_path} bulunamadı!")
        return

    # Veriyi yükle
    df = pd.read_csv(raw_csv_path, sep=',')
    df.columns = df.columns.str.strip()
    raw_data = df[["emg1", "pzt1", "pzt2"]].values.astype(np.int16)

    fs = 500

    # 1. Ham Sinyal (Float'a çevirip DC offset alalım ki grafik düzgün çıksın)
    raw_sig = raw_data.astype(np.float32) * (3.3 / 4095.0)
    raw_sig = raw_sig - np.mean(raw_sig, axis=0, keepdims=True)
    raw_emg = raw_sig[:, 0]  # Sadece EMG1

    # 2. Filtrelenmiş Sinyal (Bizim preprocess fonksiyonu)
    filtered_data = preprocess(raw_data, fs=fs)
    filtered_emg = filtered_data[:, 0]

    # Grafikleri Çiz
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("50Hz Şebeke Gürültüsü Filtreleme Analizi (EMG1)", fontsize=16)

    time_axis = np.arange(len(raw_emg)) / fs

    # --- Sol Üst: Ham Sinyal (Zaman) ---
    axs[0, 0].plot(time_axis, raw_emg, color='red', alpha=0.7)
    axs[0, 0].set_title("Ham Sinyal (Şarja Takılı - Gürültülü)")
    axs[0, 0].set_xlabel("Zaman (s)")
    axs[0, 0].set_ylabel("Voltaj (V)")

    # --- Sağ Üst: Filtreli Sinyal (Zaman) ---
    axs[0, 1].plot(time_axis, filtered_emg, color='green', alpha=0.8)
    axs[0, 1].set_title("Filtrelenmiş Sinyal (Notch + Bandpass)")
    axs[0, 1].set_xlabel("Zaman (s)")
    axs[0, 1].set_ylabel("Voltaj (V)")

    # --- Sol Alt: Ham Sinyal Spektrumu (Frekans) ---
    f_raw, Pxx_raw = welch(raw_emg, fs, nperseg=1024)
    axs[1, 0].semilogy(f_raw, Pxx_raw, color='red')
    axs[1, 0].set_title("Ham Sinyal Frekans Spektrumu")
    axs[1, 0].set_xlabel("Frekans (Hz)")
    axs[1, 0].set_ylabel("Güç")
    axs[1, 0].axvline(x=50, color='black', linestyle='--', label='50Hz (Şebeke)')
    axs[1, 0].legend()

    # --- Sağ Alt: Filtreli Sinyal Spektrumu (Frekans) ---
    f_filt, Pxx_filt = welch(filtered_emg, fs, nperseg=1024)
    axs[1, 1].semilogy(f_filt, Pxx_filt, color='green')
    axs[1, 1].set_title("Filtrelenmiş Frekans Spektrumu")
    axs[1, 1].set_xlabel("Frekans (Hz)")
    axs[1, 1].set_ylabel("Güç")
    axs[1, 1].axvline(x=50, color='black', linestyle='--', label='50Hz Çentik (Vuruldu)')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_mains_noise()