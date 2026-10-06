import numpy as np
from collections import deque
from typing import List, Optional, Dict


class Segment:
    def __init__(self, data: np.ndarray, start_idx: int, end_idx: int):
        self.data = data
        self.start_idx = start_idx
        self.end_idx = end_idx


class VAD:
    def __init__(self, config: Dict):
        self.config = config
        self.state = "SILENCE"

        # State Değişkenleri
        self.threshold = None
        self.buffer = []
        self.silence_counter = 0
        self.current_sample_idx = 0
        self.segment_start_idx = 0

        # Pre-roll Ring Buffer (Son N sample'ı her zaman tutar)
        self.preroll_ring = deque(maxlen=config["preroll_samples"])

        # Envelope filtresi için state (5Hz Lowpass IIR - EMA yaklaşımı)
        w = 2 * np.pi * config["envelope_lowpass_hz"] / config["fs"]
        self.alpha = np.cos(w) - 1 + np.sqrt(np.cos(w) ** 2 - 4 * np.cos(w) + 3)
        self.envelope_state = 0.0

    def _compute_envelope_sample(self, emg_val: float) -> float:
        """Tek bir sample için causal lowpass envelope hesaplar (Sadece 1 EMG kanalı ile)."""
        # Rectified value of the single EMG channel
        rectified_val = abs(emg_val)
        # IIR Lowpass (EMA)
        self.envelope_state = self.alpha * rectified_val + (1 - self.alpha) * self.envelope_state
        return self.envelope_state

    def calibrate(self, baseline_signal_20s: np.ndarray):
        """
        20 saniyelik başlangıç sessizlik periyodundan threshold hesaplar.
        baseline_signal_20s: (N, 5) boyutunda numpy array
        """
        envelopes = []
        for sample in baseline_signal_20s:
            # Sadece 0. kanal (EMG1) envelope hesabına gönderiliyor
            env = self._compute_envelope_sample(sample[0])
            envelopes.append(env)

        env_array = np.array(envelopes)
        self.threshold = env_array.mean() + self.config["threshold_k"] * env_array.std()
        print(f"VAD Kalibrasyonu Tamamlandı. Threshold: {self.threshold:.4f}")

    def step(self, sample_5ch: np.ndarray) -> Optional[Segment]:
        """
        Gelen tek bir (5,) boyutlu sample'ı işler ve state machine'i günceller.
        Kelime tespiti tamamlandığında Segment objesi döner.
        """
        if self.threshold is None:
            raise ValueError("VAD henüz kalibre edilmedi. Önce calibrate() çağrılmalı.")

        # 1. Envelope güncelle (Sadece 0. kanal EMG olduğu için)
        env = self._compute_envelope_sample(sample_5ch[0])
        is_active = env > self.threshold

        # 2. Pre-roll buffer'ı her zaman güncelle
        self.preroll_ring.append(sample_5ch.copy())

        self.current_sample_idx += 1
        result_segment = None

        # 3. State Machine
        if self.state == "SILENCE":
            if is_active:
                self.state = "SPEAKING"
                # Pre-roll içindeki geçmiş veriyi buffer'a aktar (Kelimenin başı kesilmesin diye)
                self.buffer = list(self.preroll_ring)
                self.segment_start_idx = self.current_sample_idx - len(self.buffer)

        elif self.state == "SPEAKING":
            self.buffer.append(sample_5ch.copy())

            if not is_active:
                self.state = "TAILING_OFF"
                self.silence_counter = 0
            elif len(self.buffer) >= self.config["max_duration_samples"]:
                # Kelime çok uzun sürdü, zorla kes
                result_segment = self._finalize_segment()
                self.state = "SILENCE"

        elif self.state == "TAILING_OFF":
            self.buffer.append(sample_5ch.copy())

            if is_active:
                # Sessizlik bitti, demek ki kelime devam ediyormuş
                self.state = "SPEAKING"
                self.silence_counter = 0
            else:
                self.silence_counter += 1
                if self.silence_counter >= self.config["silence_gap_samples"]:
                    # Sessizlik yeterince uzun sürdü, segmentasyon kararını ver
                    if len(self.buffer) >= self.config["min_duration_samples"]:
                        result_segment = self._finalize_segment()
                    else:
                        # Spurious (yanlış) tetiklenme, çok kısa
                        self.buffer = []
                    self.state = "SILENCE"

        return result_segment

    def _finalize_segment(self) -> Segment:
        """Buffer'daki veriyi sabit uzunluğa (pad/crop) getirir ve Segment döner."""
        data_array = np.array(self.buffer)
        target_len = self.config["fixed_window_samples"]

        if len(data_array) < target_len:
            # Sağdan sıfırlarla pad et
            pad_width = target_len - len(data_array)
            # Sadece zaman ekseninde padding yap (0. eksen), kanal ekseninde (1. eksen) yapma
            data_array = np.pad(data_array, ((0, pad_width), (0, 0)), mode='constant')
        elif len(data_array) > target_len:
            # Ortadan kırp (Baştan ve sondan eşit at)
            excess = len(data_array) - target_len
            start_trim = excess // 2
            data_array = data_array[start_trim: start_trim + target_len]

        seg = Segment(data_array, self.segment_start_idx, self.segment_start_idx + len(self.buffer))
        self.buffer = []
        return seg

    def process_offline(self, full_signal_5ch: np.ndarray) -> List[Segment]:
        """Eğitim verisi için sarmalayıcı. step() fonksiyonunu for döngüsüne sokar."""
        segments = []
        for sample in full_signal_5ch:
            seg = self.step(sample)
            if seg is not None:
                segments.append(seg)
        return segments


# Kullanım Sabitleri (Yiğit'in Kotlin'de okuyabilmesi için JSON olarak export edilecek)
VAD_CONFIG = {
    "fs": 500,
    "threshold_k": 3.0,
    "min_duration_samples": 150,  # 300 ms
    "max_duration_samples": 1000,  # 2 saniye
    "silence_gap_samples": 100,  # 200 ms bekleme
    "preroll_samples": 50,  # 100 ms öncesini al
    "envelope_lowpass_hz": 5.0,
    "fixed_window_samples": 1000,  # 1D-CNN her zaman 2s (1000 sample) alır
}