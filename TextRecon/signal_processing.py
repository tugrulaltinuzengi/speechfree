import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import json


# ==========================================
# 1. Filtre Katsayılarını Hesaplama ve Kaydetme
# ==========================================
def get_butter_bandpass_coeffs(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def get_notch_coeffs(freq, q, fs):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return b, a


def export_filter_configs(filepath="preprocessing_params.json", fs=500):
    """
    Filtre katsayılarını Yiğit'in Kotlin'de kullanabilmesi için JSON olarak dışa aktarır.
    """
    b_emg, a_emg = get_butter_bandpass_coeffs(20, 200, fs, order=4)
    b_pzt, a_pzt = get_butter_bandpass_coeffs(5, 200, fs, order=4)
    b_notch, a_notch = get_notch_coeffs(50, 30, fs)

    config = {
        "fs": fs,
        "emg_bandpass": {"b": b_emg.tolist(), "a": a_emg.tolist()},
        "pzt_bandpass": {"b": b_pzt.tolist(), "a": a_pzt.tolist()},
        "notch_50hz": {"b": b_notch.tolist(), "a": a_notch.tolist()}
    }

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Filtre konfigürasyonu {filepath} adresine kaydedildi.")


# ==========================================
# 2. Causal Sinyal İşleme Pipeline'ı
# ==========================================
def preprocess(raw_5ch, fs=500):
    """
    raw_5ch: shape (N_samples, 5), int16 ADC değerleri
    return: shape (N_samples, 5), float32 normalize edilmiş sinyal

    Güncellenmiş Kanal Sırası:
    0: EMG1, 1: PZT1, 2: PZT2, 3: PZT3, 4: PZT4
    """
    # 1. Float'a çevir ve Voltaja ölçekle (0-4095 ADC -> 0-3.3V)
    sig = raw_5ch.astype(np.float32) * (3.3 / 4095.0)

    # 2. DC offset çıkar
    sig = sig - np.mean(sig, axis=0, keepdims=True)

    # Katsayıları al
    b_emg, a_emg = get_butter_bandpass_coeffs(20, 200, fs, order=4)
    b_pzt, a_pzt = get_butter_bandpass_coeffs(5, 200, fs, order=4)
    b_notch, a_notch = get_notch_coeffs(50, 30, fs)

    processed_sig = np.zeros_like(sig)

    # 3. Band-pass filtre ve 4. 50Hz Notch
    # Döngü 5 kanala çekildi
    for i in range(5):
        # Filtre tipini belirle
        if i < 1:
            # Sadece ilk kanal (0) EMG
            filtered_bp = lfilter(b_emg, a_emg, sig[:, i])
        else:
            # Geri kalanlar (1, 2, 3, 4) PZT
            filtered_bp = lfilter(b_pzt, a_pzt, sig[:, i])

        # Notch filtresini uygula ve kaydet
        processed_sig[:, i] = lfilter(b_notch, a_notch, filtered_bp)

    return processed_sig