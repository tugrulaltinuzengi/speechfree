# Kayit Sistemi Kullanim Klavuzu

İki dosya:
- `ESP32_Electrolarynx_Recorder.ino` — ESP32-S3 firmware'i
- `DEMO_VAD_6ch.m` — MATLAB kayit scripti (25 tekrar protokolu)

---

## 1. Donanim Hazirligi

### 1.1 ESP32-S3 Pin Bağlantıları

Sensor çıkışlarını şu GPIO'lara bağla:

| GPIO | Sensor | Yerleşim |
|------|--------|----------|
| GPIO 1 | EMG1 çıkışı | Submental sol |
| GPIO 2 | EMG2 çıkışı | Submental sağ |
| GPIO 3 | PZT1 (buffer çıkışı) | Submental orta hat |
| GPIO 4 | PZT2 (buffer çıkışı) | Submandibular sol |
| GPIO 5 | PZT3 (buffer çıkışı) | Submandibular sağ |
| GPIO 6 | PZT4 (buffer çıkışı) | Lateral boyun (sol SCM) |

GND ortak. EMG modüllerini ESP32'nin 3.3V'undan değil **harici 5V** veya bataryadan besle (gürültü izolasyonu için). Op-amp buffer'ları 3.3V'tan beslenebilir.

### 1.2 Piezo Buffer Devresi (her piezo için)

Her piezonun ESP32'ye bağlanmadan önce bir **op-amp voltage follower**'dan geçmesi gerekiyor. TL072 veya benzeri JFET-input op-amp:

```
Piezo (+) ──┬── 1MΩ ── 1.65V (DC bias, 3.3V'tan iki 100k direnç ile)
            │
            ├── 100nF ── Op-amp(+)
            │
Piezo (-) ──┴── GND

Op-amp çıkışı (V_out=V_in, gain=1) → ESP32 GPIO
```

Buffer olmazsa piezo sinyali ADC tarafından "yenir", anlamlı sinyal göremezsin.

---

## 2. Firmware Yükleme

### Arduino IDE Kurulumu
1. Arduino IDE 2.x kur
2. **Boards Manager**: "esp32" by Espressif Systems → kur (en güncel versiyon)
3. Board seç: **Tools → Board → ESP32S3 Dev Module**
4. Port seç: **Tools → Port → COMx** (Windows) veya `/dev/tty.usbserial-...` (Mac/Linux)
5. **USB CDC On Boot: Enabled** olarak ayarla (önemli, yoksa Serial göremezsin)
6. **Upload Speed: 921600**

### Yükleme
1. `ESP32_Electrolarynx_Recorder.ino` dosyasını aç
2. Upload butonu (sağ ok)
3. Yükleme bittiğinde: **Tools → Serial Monitor**, baud **921600** seç
4. Görmek istediğin:
   ```
   # ESP32 Electrolarynx Recorder v1.0
   # 6 channels @ 500 Hz, 12-bit ADC1
   # Commands: START, STOP, MARK:<str>, PING, STATUS
   READY
   ```

### Hızlı Test
Serial Monitor'da şunları yaz (alttaki giriş kutusu, "newline" seçili olmalı):
- `PING` → `PONG` dönmeli
- `START` → 6 kanal × 500 Hz CSV akışı başlamalı (saniyede 500 satır)
- `STOP` → akış durmalı

Akış sırasında çıkan satırlar şöyle görünmeli:
```
2000,2048,2050,1832,1845,1820,1855
4000,2049,2051,1830,1846,1821,1856
...
```

İlk sütun timestamp (mikrosaniye), sonraki 6 sütun ham ADC değeri (0-4095). Sayılar tamamen 0 veya 4095'e sabitlenmişse → bağlantı problemi var.

---

## 3. Kayit Almak

### MATLAB Hazırlığı
1. MATLAB R2020b veya üstü (serialport API için)
2. **Instrument Control Toolbox** kurulu olmalı
3. ESP32'nin görüldüğü COM portunu öğren:
   - Windows: Device Manager → Ports (COM & LPT)
   - Mac/Linux: terminal'de `ls /dev/tty.*` veya `ls /dev/ttyUSB*`

### Script Ayarı
`DEMO_VAD_6ch.m` dosyasını aç, en üstteki **AYARLAR** bölümünü güncelle:

```matlab
SERIAL_PORT = "COM4";        % Senin portuna göre
SUBJECT_ID  = "S01";         % Bu oturumun denek ID'si
OPERATOR    = "Serhat";
```

Diğer parametreler (25 tekrar, 9-17. tekrarlarda mola, vs.) standart — değiştirme.

### Oturum Akışı
1. ESP32'yi USB ile bağla (Arduino IDE Serial Monitor'ü kapat — port kilitli kalmasın)
2. Sensörleri **PROJECT_OVERVIEW.md**'deki sensor layout'a göre yerleştir
3. MATLAB'da `DEMO_VAD_6ch` çalıştır (Run veya `>> DEMO_VAD_6ch`)
4. **Pre-flight checklist**'i tek tek onayla (her madde için Enter)
5. Script otomatik:
   - 20s baseline başlatır
   - 25 tekrar × 10 kelime gösterir (rastgele sırada)
   - 9. ve 17. tekrardan sonra 3 dakika mola verir
   - 20s tail kaydeder
6. Bittiğinde "Oturuma dair notlar" sorar — anormal bir şey varsa yaz

**Tahmini toplam süre:** 40-45 dakika (molalar dahil)

### Oturum Sırasında Kurallar
- Konuşma normal sesli olacak (hasta gibi sessiz değil)
- Kelimeyi penceredeki 5 saniye içinde söyle, doğal ritimle, acele etme
- Yanlış söyledin sandıysan **endişelenme**, devam et — VAD offline'da segmentleri ayıklayacak, birkaç bozuk tekrar problem olmaz
- Mola sırasında elektrotlara dokunma, kabloları çekme

---

## 4. Çıktı Dosyaları

Her oturum şu klasör yapısını oluşturur:

```
recordings/
└── session_S01_2026-04-25_14-30/
    ├── raw_6ch.csv         # Ham 6-kanal veri, 500 Hz, ~50 MB civarı
    ├── markers.csv         # Talimat ve pencere zamanları
    └── session_meta.json   # Oturum metadata
```

Bu klasörü olduğu gibi Mert'e gönder (paylaşımlı sürücü, GitHub LFS, vb.).

### raw_6ch.csv örneği
```
timestamp_us,emg1,emg2,pzt1,pzt2,pzt3,pzt4
2000,2048,2050,1832,1845,1820,1855
4000,2049,2051,1830,1846,1821,1856
...
```

### markers.csv örneği
```
timestamp_us,event,word,repetition
0,baseline_start,,0
20000000,baseline_end,,0
20100000,instruction,EVET,1
22100000,window_start,EVET,1
27100000,window_end,EVET,1
...
```

---

## 5. Sık Karşılaşılan Problemler

| Problem | Çözüm |
|---------|-------|
| MATLAB "Port not found" hatası | Arduino IDE Serial Monitor açık olabilir, kapat. COM port doğru mu kontrol et. |
| ESP32 bağlanıyor ama veri yok | Serial Monitor'de manuel `START` yaz, çıkış var mı bak. Yoksa firmware doğru yüklenmemiş. |
| Sample rate 500 Hz değil (eksik satır) | Baud rate 921600 olduğundan emin ol. USB kablosunun data desteği var mı (bazıları sadece şarj). |
| Kanallardan biri sürekli 0 veya 4095 | O GPIO'ya bağlantı kopuk veya kısa devre. EMG modülünden çıkış geliyor mu osilaskop/voltmetre ile bak. |
| Tüm kanallarda 50 Hz dalga | Topraklama problemi. ESP32'yi laptop yerine bataryadan besle. Referans elektrodu yenile. |
| MATLAB callback satır kaçırıyor | Script çalıştırırken laptop başka ağır iş yapmasın. Diğer programları kapat. |
| Piezolardan sinyal yok | Buffer devresi problemi. Op-amp besleme var mı, DC bias 1.65V'a oturdu mu? |
| Mola sırasında kayıt akıyor | Bu **doğru** — mola da kayıt edilir, sonra Mert filtreler. Endişelenme. |

---

## 6. Pilot Oturum Notu

İlk oturum **pilot** — Mert'e gönder, "veri kalitesi yeterli mi?" diye onay al. Onay gelmeden 6+ ana oturuma başlama. Mert'in pilot raporundan sonra (ör. "PZT4'te 50 Hz baskın, topraklama düzelt") düzeltmeleri yap, sonra ana oturumlara geç.

---

## 7. Bağlantı / Pinout Hızlı Referans

```
ESP32-S3 DevKit
┌─────────────────┐
│                 │
│  GPIO1 ←  EMG1  │  Submental sol
│  GPIO2 ←  EMG2  │  Submental sağ
│  GPIO3 ←  PZT1  │  (buffer'dan) Submental orta
│  GPIO4 ←  PZT2  │  (buffer'dan) Submandibular sol
│  GPIO5 ←  PZT3  │  (buffer'dan) Submandibular sağ
│  GPIO6 ←  PZT4  │  (buffer'dan) Lateral boyun sol
│                 │
│  GND   ←  Ortak GND (tüm sensörler + buffer'lar)
│  3V3   →  Op-amp besleme
│  USB   →  Laptop/PC
│                 │
└─────────────────┘
```

Sorularda Yiğit'e veya Mert'e sor.
