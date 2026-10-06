/*
 * ESP32_Electrolarynx_Recorder.ino
 * * 6-kanal kayit firmware'i (sadece USB serial, kayit fazi icin).
 * * Kanallar (Yeni Pinout'a göre güncellendi):
 * ADC1_CH0 (GPIO36 / VP) - EMG1 - Submental sol
 * ADC1_CH3 (GPIO39 / VN) - EMG2 - Submental sag
 * ADC1_CH6 (GPIO34)      - PZT1 - Submental orta hat
 * ADC1_CH7 (GPIO35)      - PZT2 - Submandibular sol
 * ADC1_CH4 (GPIO32)      - PZT3 - Submandibular sag
 * ADC1_CH5 (GPIO33)      - PZT4 - Lateral boyun (SCM)
 * * Sampling rate: 500 Hz (sıkı timing, hardware timer ile)
 * ADC: 12-bit, 0-3.3V (ADC1, BLE/WiFi cakismasi yok)
 * * Cikti formati (ASCII CSV, her satir bir sample):
 * timestamp_us,ch0,ch1,ch2,ch3,ch4,ch5
 * * Komutlar (Serial uzerinden, newline ile sonlandirilmis):
 * START      - Sampling baslat (timestamp 0'dan baslar)
 * STOP       - Sampling durdur
 * MARK:<str> - Marker satiri yaz (kayit protokolu icin)
 * PING       - Hayatta misin testi (PONG donderir)
 * STATUS     - Sampling durumunu raporla
 * * Baud rate: 921600 (yuksek throughput icin)
 */

#include <Arduino.h>
#include "driver/adc.h"
#include "esp_adc_cal.h"

// ====== Konfigurasyon ======
#define SAMPLE_RATE_HZ    500
#define SAMPLE_PERIOD_US  (1000000 / SAMPLE_RATE_HZ)  // 2000 us
#define SERIAL_BAUD       921600
#define NUM_CHANNELS      6

// ADC kanallari (ADC1) - YENİ BAĞLANTILARA GÖRE DÜZENLENDİ
const adc1_channel_t ADC_CHANNELS[NUM_CHANNELS] = {
  ADC1_CHANNEL_0,  // GPIO25 (VP) - EMG1
  ADC1_CHANNEL_3,  // GPIO39 (VN) - EMG2
  ADC1_CHANNEL_6,  // GPIO34      - PZT1
  ADC1_CHANNEL_7,  // GPIO35      - PZT2
  ADC1_CHANNEL_4,  // GPIO32      - PZT3
  ADC1_CHANNEL_5,  // GPIO33      - PZT4
};

// ====== Global durum ======
hw_timer_t* sampleTimer = NULL;
volatile bool sampleFlag = false;
volatile bool sampling = false;
uint32_t sessionStartUs = 0;

// Hardware timer ISR - sample bayragini set eder, asil okuma loop'ta
void IRAM_ATTR onSampleTimer() {
  sampleFlag = true;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 2000) { delay(10); }

  // ADC1 konfigurasyonu - 12-bit, 0-3.3V
  adc1_config_width(ADC_WIDTH_BIT_12);
  for (int i = 0; i < NUM_CHANNELS; i++) {
    adc1_config_channel_atten(ADC_CHANNELS[i], ADC_ATTEN_DB_11);
  }

  // Hardware timer - 1 MHz tick (ESP32 Core 3.x API Güncellemesi)
  sampleTimer = timerBegin(1000000); 
  timerAttachInterrupt(sampleTimer, &onSampleTimer);
  timerAlarm(sampleTimer, SAMPLE_PERIOD_US, true, 0); 
  timerStop(sampleTimer); 

  Serial.println("# ESP32 Electrolarynx Recorder v1.0");
  Serial.println("# 6 channels @ 500 Hz, 12-bit ADC1");
  Serial.println("# Commands: START, STOP, MARK:<str>, PING, STATUS");
  Serial.println("READY");
}

void startSampling() {
  if (sampling) return;
  sessionStartUs = micros();
  sampleFlag = false;
  sampling = true;
  timerStart(sampleTimer); 
  Serial.println("# SAMPLING_STARTED");
}

void stopSampling() {
  if (!sampling) return;
  timerStop(sampleTimer); 
  sampling = false;
  Serial.println("# SAMPLING_STOPPED");
}

void handleCommand(const String& cmd) {
  if (cmd == "START") {
    startSampling();
  }
  else if (cmd == "STOP") {
    stopSampling();
  }
  else if (cmd.startsWith("MARK:")) {
    // Marker'i timestamp ile birlikte yaz, # ile baslat ki parser ayirsin
    uint32_t t = sampling ? (micros() - sessionStartUs) : 0;
    Serial.print("# MARKER,");
    Serial.print(t);
    Serial.print(",");
    Serial.println(cmd.substring(5));
  }
  else if (cmd == "PING") {
    Serial.println("PONG");
  }
  else if (cmd == "STATUS") {
    Serial.print("# STATUS sampling=");
    Serial.print(sampling ? "1" : "0");
    Serial.print(" rate=");
    Serial.print(SAMPLE_RATE_HZ);
    Serial.print(" channels=");
    Serial.println(NUM_CHANNELS);
  }
  else if (cmd.length() > 0) {
    Serial.print("# UNKNOWN_CMD: ");
    Serial.println(cmd);
  }
}

void loop() {
  // 1) Serial komut isle
  while (Serial.available()) {
    static String buf = "";
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (buf.length() > 0) {
        handleCommand(buf);
        buf = "";
      }
    } else {
      buf += c;
      if (buf.length() > 128) buf = "";  // overflow koruma
    }
  }

  // 2) Sample alma
  if (sampling && sampleFlag) {
    sampleFlag = false;
    uint32_t t = micros() - sessionStartUs;

    // 6 kanali sirayla oku (ADC1, hizli)
    int v[NUM_CHANNELS];
    for (int i = 0; i < NUM_CHANNELS; i++) {
      v[i] = adc1_get_raw(ADC_CHANNELS[i]);
    }

    // Tek satirda yaz (CSV format)
    // 921600 baud'da 6 kanal + timestamp ~50 byte * 500 Hz = 25 kB/s, rahat
    Serial.print(t);
    Serial.print(',');
    Serial.print(v[0]); Serial.print(',');
    Serial.print(v[1]); Serial.print(',');
    Serial.print(v[2]); Serial.print(',');
    Serial.print(v[3]); Serial.print(',');
    Serial.print(v[4]); Serial.print(',');
    Serial.println(v[5]);
  }
}