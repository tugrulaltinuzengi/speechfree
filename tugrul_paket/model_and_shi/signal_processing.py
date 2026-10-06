import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import json


def get_butter_bandpass_coeffs(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')


def get_notch_coeffs(freq, q, fs):
    return iirnotch(freq / (0.5 * fs), q)


def export_filter_configs(filepath="preprocessing_params.json", fs=500):
    b_emg, a_emg = get_butter_bandpass_coeffs(20, 200, fs, order=4)
    b_pzt, a_pzt = get_butter_bandpass_coeffs(5, 200, fs, order=4)
    b_notch, a_notch = get_notch_coeffs(50, 30, fs)
    config = {
        "fs": fs,
        "channel_order": ["emg1", "pzt1", "pzt2"],
        "emg_bandpass": {"b": b_emg.tolist(), "a": a_emg.tolist()},
        "pzt_bandpass": {"b": b_pzt.tolist(), "a": a_pzt.tolist()},
        "notch_50hz": {"b": b_notch.tolist(), "a": a_notch.tolist()},
    }
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)


def preprocess(raw_3ch, fs=500, adc_max=1023, vref=3.3):
    """
    raw_3ch: (N, 3) int -- kanal sırası [emg1, pzt1, pzt2]
    PoC: 10-bit ADC (adc_max=1023). Yeni donanım: 12-bit (adc_max=4095).
    """
    sig = raw_3ch.astype(np.float32) * (vref / float(adc_max))
    sig = sig - np.mean(sig, axis=0, keepdims=True)

    b_emg, a_emg = get_butter_bandpass_coeffs(20, 200, fs, order=4)
    b_pzt, a_pzt = get_butter_bandpass_coeffs(5, 200, fs, order=4)
    b_notch, a_notch = get_notch_coeffs(50, 30, fs)

    out = np.zeros_like(sig)
    for i in range(3):
        bp = lfilter(b_emg if i == 0 else b_pzt, a_emg if i == 0 else a_pzt, sig[:, i])
        out[:, i] = lfilter(b_notch, a_notch, bp)
    return out
