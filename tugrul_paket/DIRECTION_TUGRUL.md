# Tuğrul — VAD Karşılaştırma Görevi

Selam,

Elimizdeki 4 kayıttan oluşan veri setini **mevcut EMG-tabanlı VAD'ımızla** yeniden segmente edip, **kayıt sırasında işaretlenen manuel REP marker'larıyla** karşılaştırmanı istiyorum. Amaç: VAD'ın onset/offset tespitini referans marker'lara göre nicel olarak değerlendirmek ve sınıflandırıcıyı yeni segmentlerle yeniden eğittiğimizde accuracy'de ne değişiyor görmek.

Pakette hem ham kayıt verisi hem de mevcut VAD/model kodunun tamamı var — kanal mevzularını 3-kanala göre düzelttim, hazır.

---

## 1. Paket içeriği

```
tugrul_paket/
├── DIRECTION_TUGRUL.md           ← bu dosya
├── recordings/                    ← 4 kayıt (sensör + ses + marker)
│   ├── kayit1_ALL_SENSORS.csv     500 Hz, kolonlar: timestamp_s, pzt1, pzt2, emg1
│   ├── kayit1_MARKERS.csv         time_s, label  (manuel marker'lar)
│   ├── kayit1_SES.wav             44.1 kHz mono, sensörlerle senkron (t=0 ortak)
│   ├── kayit2_*  ... kayit4_*
└── model_and_shi/                 ← VAD + model kodu (3 kanala düşürülmüş)
    ├── vad.py                     ← EMG envelope + state-machine VAD
    ├── signal_processing.py       ← bandpass + 50Hz notch (3 kanal)
    ├── build_dataset.py           ← VAD ile session'ları kesip dataset.npz üretir
    ├── models.py                  ← 1D-CNN ve LSTM (in_channels=3)
    ├── train_base_model.py        ← base model eğitimi
    ├── base_model.pth             ← mevcut eğitilmiş base model
    ├── dataset.npz                ← önceki segment'lerle üretilmiş dataset
    ├── evaluation.py, ...         ← test/değerlendirme yardımcıları
    └── ...
```

### Sensör verisi ayrıntıları

- **Örnekleme:** 500 Hz (timestamp adımı ~2 ms)
- **Kanallar:** 3 — `pzt1`, `pzt2`, `emg1`
- **CSV kolon sırası:** `timestamp_s, pzt1, pzt2, emg1` ← dikkat, kod `[emg1, pzt1, pzt2]` sırası bekliyor (aşağıda)
- **Süre:** ~350 sn / kayıt
- **ADC:** 10-bit (0–1023). Eğer yeni donanım 12-bit ise `preprocess(raw, fs=500, adc_max=4095)` ile çağır.

### Marker formatı

```
time_s, label
0.0001888, SESSIZLIK_BASLANGIC
14.2865076, KELIME_EVET_BASLANGIC
14.287535, EVET_REP1
17.2940321, EVET_REP2
...
44.3213723, KELIME_EVET_BITIS
46.9037877, KELIME_HAYIR_BASLANGIC
46.9040474, HAYIR_REP1
...
341.4004723, KELIME_GUNAYDIN_BITIS
341.400699, SESSIZLIK_BITIS
```

10 kelime (EVET, HAYIR, MERHABA, TESEKKURLER, SU, YARDIM, TAMAM, DUR, GEL, GUNAYDIN), her biri 10 tekrar, kelime-arası ~3 sn aralık. **Bu marker'lar onset zamanlarıdır, offset yok.** Referans segment çıkarırken offset'i bir sonraki REP'in 200 ms öncesi (veya kelimenin son REP'i ise `KELIME_X_BITIS`) olarak tahmin et.

---

## 2. Yapılması gereken

### Adım 0 — Veriyi VAD'ın beklediği formata çevir

Kod 3 kanalı `[emg1, pzt1, pzt2]` sırasında bekliyor; bizim CSV'lerde sıra `pzt1, pzt2, emg1`. Yükleyici loader'ı şöyle yaz:

```python
import pandas as pd, numpy as np
df = pd.read_csv("recordings/kayit1_ALL_SENSORS.csv")
raw = df[["emg1", "pzt1", "pzt2"]].values.astype(np.int16)  # permute burada
```

Sonra `preprocess(raw, fs=500, adc_max=1023)` ile filtreleme + `VAD(VAD_CONFIG)` ile segmentasyon.

### Adım 1 — Yeni VAD ile segmente et

Her kayıt için:

```python
from signal_processing import preprocess
from vad import VAD, VAD_CONFIG

processed = preprocess(raw, fs=500, adc_max=1023)
# İlk 14 saniyede sessizlik var (ilk REP ~14.3 sn'de) — kalibrasyonu burdan al
calibration_samples = int(13 * 500)  # 13 sn güvenli
vad = VAD(VAD_CONFIG)
vad.calibrate(processed[:calibration_samples])
segments = vad.process_offline(processed[calibration_samples:])
```

`segments` listesi `Segment(data, start_idx, end_idx)` objeleri verir. `start_idx`/`end_idx`, `calibration_samples`'ten sonraki indeksler — global zamana çevirmek için: `t = (calibration_samples + idx) / 500.0`.

> **Not:** `VAD_CONFIG`'deki `threshold_k=3.0`'ı 4 kayıt üzerinde optimize edebilirsin — özellikle PZT'nin EMG'ye sızdığı kayıtlarda 2.5 daha iyi olabilir. Ama final raporda hangi `k`'yı kullandığını ve bunu nasıl seçtiğini yaz.

### Adım 2 — Marker'lardan referans segment çıkar

```
EVET_REP1: onset = 14.287, offset = (EVET_REP2 onset) - 0.2 = 17.094
EVET_REP10: offset = KELIME_EVET_BITIS = 44.321
```

Tüm REP'ler için `(label, onset_s, offset_s)` listesi üret. Toplam ~100 REP / kayıt × 4 kayıt = 400 referans segment beklenir.

### Adım 3 — Hizalama metrikleri

VAD segmentlerini referans REP segmentleriyle eşle (IoU > 0.3 ile match). Her kayıt için ayrı, sonra hepsi birleşik:

| Metrik | Açıklama |
|---|---|
| `match_rate` | Bir VAD segmentine eşleşen REP / toplam REP |
| `false_positive_count` | Hiçbir REP'le eşleşmeyen VAD segmenti (sessizlikteki tetiklemeler) |
| `miss_count` | Hiçbir VAD segmentine eşleşmeyen REP |
| `Δonset` | `vad_onset − rep_onset`, ms cinsinden mean / std / MAE |
| `Δoffset` | `vad_offset − rep_offset_tahmin`, ms cinsinden mean / std / MAE |
| `IoU` | Eşleşen çiftler için ortalama |

### Adım 4 — Sensör SNR sanity check

Her VAD segmenti için sEMG RMS hesapla, ilk 13 sn sessizliğin RMS'iyle kıyasla. SNR < 3 dB olan segmentleri ayrı işaretle (`low_snr=True` kolonu) — sınıflandırıcı bu segmentleri muhtemelen yanlış sınıflar.

### Adım 5 — Sınıflandırıcıyı yeni segmentlerle yeniden eğit

`build_dataset.py`'i bizim kayıt formatımıza uyarla (şu an `recordings/session_*/raw_3ch.csv` + `markers.csv` bekliyor; bizim formatımız `kayitN_ALL_SENSORS.csv` + `kayitN_MARKERS.csv`). Bunu yeni bir `build_dataset_recordings.py` olarak yaz, eski dosyayı bozma. Etiketleme için marker'lardan REP'leri sırayla VAD segmentlerine eşle (zamansal sıra).

Sonra `train_base_model.py` ile within-session ve cross-session eğitimi yap. **Karşılaştırma:** mevcut sonuçlarımız within-session %67 Top-1 / %87 Top-3, cross-session en iyi %52 Top-1. Yeni VAD ile bu sayılar nasıl değişiyor?

### Adım 6 — Çıktılar

1. **`kayitN_VAD_vs_MARKER.csv`** her kayıt için:
   `rep_label, rep_onset, rep_offset, vad_onset, vad_offset, delta_onset_ms, delta_offset_ms, iou, semg_rms, snr_db, low_snr, matched`

2. **`summary.csv`** — 4 kayıt birleşik, kelime bazında match rate / mean Δonset / FP count.

3. **Plotlar:**
   - Δonset histogramı (hepsi birleşik)
   - Her kayıt için timeline overlay: x ekseni zaman, üstte EMG envelope + threshold line, altta yeşil bantlar VAD segmentleri, kırmızı dikey çizgiler REP marker'ları
   - Confusion matrix: yeni VAD segmentleriyle eğitilen modelin within-session performansı

4. **Kısa rapor (`vad_compare_report.md`):** threshold seçimi, metrikler tablosu, eski vs yeni accuracy karşılaştırması, gözlemlediğin failure mode'lar (örn: çok hızlı tekrarlarda VAD birleştiriyor mu, sessizlikte FP var mı).

### Adım 7 — Repo

Hepsini `Azafuse/electrolarynx-capstone` (veya aktif repo neyse) altında `vad_compare/` klasörüne commit'le. README'de özet sonuçlar.

---

## 3. Uyarılar / gotcha'lar

- **CSV kolon sırası:** verideki `pzt1, pzt2, emg1` ≠ kodun beklediği `emg1, pzt1, pzt2`. Permute etmeyi unutma, yoksa VAD threshold'u yanlış kanaldan hesaplar ve hiç tetiklenmez.
- **ADC bit derinliği:** kayıtlar 10-bit görünüyor (0–1023 aralığında). Eğer max değer 1023'ü aşarsa donanım değişmiş demektir, `adc_max=4095` ver.
- **Senkron varsayımı:** sensör CSV ve WAV `t=0`'da hizalı kabul ediliyor. Bir kayıtta görsel olarak doğrula (sessizlik bölgesi + ilk REP onset). Eğer drift varsa hizalama değeri hesapla, raporda yaz.
- **Marker'larda offset yok**, sadece onset. Yukarıdaki "bir sonraki REP'in 200 ms öncesi" yaklaşımı tutarlı bir referans verir ama mükemmel değil — bu yüzden Δoffset metriğini Δonset'ten daha az ciddiye al.
- **VAD config sabit kalsın** kayıtlar arası karşılaştırma için. Sadece `threshold_k`'yı tune et, diğer parametreler (`min/max_duration`, `silence_gap`) sabit.
- `generate_fixtures.py` hâlâ eski 6 kanallı synthetic data üretiyor — bizim için ilgisiz, dokunma.

---

## 4. Teslim süresi

Önce 1 kayıt üzerinde end-to-end pipeline'ı çalıştır, plot'u bana at — onay verince 4 kayıt + retraining'e geç. Bu ilk pass için ~2 gün, full karşılaştırma + retraining için 1 hafta hedefliyoruz.

Soru olursa Slack'ten yaz.

— Yeet
