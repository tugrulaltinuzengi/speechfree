"""Diagnose calibration period and find best VAD channel/window."""
import pandas as pd, numpy as np
from signal_processing import preprocess
from scipy.signal import lfilter
from vad import VAD_CONFIG

df = pd.read_csv('../recordings/kayit1_ALL_SENSORS.csv')
raw = df[['emg1','pzt1','pzt2']].values.astype('int16')
processed = preprocess(raw, fs=500, adc_max=1023)
FS = 500; CAL = 13*FS

w = 2*np.pi * VAD_CONFIG['envelope_lowpass_hz'] / FS
alpha = np.cos(w) - 1 + np.sqrt(np.cos(w)**2 - 4*np.cos(w) + 3)

print('=== EMG1 IIR envelope by 1-second windows during calibration ===')
for t in range(13):
    i0, i1 = t*FS, (t+1)*FS
    env = lfilter([alpha],[1.0,-(1.0-alpha)], np.abs(processed[i0:i1, 0]))
    print(f'  t={t:2d}-{t+1:2d}s  mean={env.mean():.4f}  std={env.std():.4f}  max={env.max():.4f}')

print()
print('=== RAW EMG1 stats (before preprocess) by 1s windows ===')
for t in range(13):
    i0, i1 = t*FS, (t+1)*FS
    chunk = raw[i0:i1, 0].astype(float)  # emg1 is column 0 after permutation
    print(f'  t={t:2d}-{t+1:2d}s  min={chunk.min():.0f}  max={chunk.max():.0f}  std={chunk.std():.1f}')

print()
print('=== Per-word EMG1 max envelope (all reps, seeking max k required) ===')
df_m = pd.read_csv('../recordings/kayit1_MARKERS.csv')
import re
rep_re = re.compile(r'^([A-Z]+)_REP\d+$')
env_full = lfilter([alpha],[1.0,-(1.0-alpha)], np.abs(processed[:, 0]))

# calibration stats
cal_mean = np.abs(processed[:CAL, 0]).mean()
cal_env = lfilter([alpha],[1.0,-(1.0-alpha)], np.abs(processed[:CAL, 0]))
cal_m, cal_s = cal_env.mean(), cal_env.std()
print(f'Calibration (0-13s): env_mean={cal_m:.4f}  env_std={cal_s:.4f}')

word_peaks = {}
for _, row in df_m.iterrows():
    m = rep_re.match(str(row['label']).strip())
    if m:
        word = m.group(1)
        t = float(row['time_s'])
        i0, i1 = int(t*FS), int((t+1.5)*FS)
        peak = env_full[i0:i1].max()
        if word not in word_peaks:
            word_peaks[word] = []
        word_peaks[word].append(peak)

print(f'{"Word":<14} {"mean_peak":>10} {"max_peak":>10} {"k_needed":>10}')
for word, peaks in word_peaks.items():
    mp = np.mean(peaks)
    mx = np.max(peaks)
    k_needed = (mx - cal_m) / cal_s if cal_s > 0 else 999
    print(f'{word:<14} {mp:>10.4f} {mx:>10.4f} {k_needed:>10.2f}')
