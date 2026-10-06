# SpeechFree TTS — Yigit Entegrasyon Kilavuzu

## Ne yapar
Mert'in EMG classifier'i bir word_index (0-9) urettugunda,
ElevenLabs API'yi cagirarak Serhat'in sesiyle o kelimeyi sentezler ve oynatir.

**Tahmini gecikme:** 550ms – 1100ms (ag + uretim + MediaPlayer)

---

## 1. Dosyayi projeye ekle

`ElevenLabsTtsPlayer.kt` dosyasini projenin uygun package'ina kopyala:

```
app/src/main/java/com/speechfree/tts/ElevenLabsTtsPlayer.kt
```

Package satirini kendi projenle eslesecek sekilde guncelle:
```kotlin
package com.senin.paket.adi
```

---

## 2. Internet izni

`AndroidManifest.xml` icinde su satirin olduguna emin ol:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

---

## 3. Kullanim

```kotlin
class MainActivity : AppCompatActivity() {

    private lateinit var tts: ElevenLabsTtsPlayer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tts = ElevenLabsTtsPlayer(this, apiKey = BuildConfig.ELEVENLABS_KEY, voiceId = BuildConfig.ELEVENLABS_VOICE)  // never commit the key
    }

    // Mert'in classifier'indan index geldiginde bu fonksiyonu cagir:
    fun onWordDetected(wordIndex: Int) {
        tts.speak(wordIndex)
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.release()
    }
}
```

---

## 4. Word Index Eslesmesi

Mert'in classifier ciktisi bu sirayla eslestirilmeli:

| Index | Kelime      |
|-------|-------------|
| 0     | EVET        |
| 1     | HAYIR       |
| 2     | MERHABA     |
| 3     | TESEKKURLER |
| 4     | SU          |
| 5     | YARDIM      |
| 6     | TAMAM       |
| 7     | DUR         |
| 8     | GEL         |
| 9     | GUNAYDIN    |

Bu siralama session_meta.json'daki `words` dizisiyle birebir ayni.

---

## 5. Ekstra dependency yok

`ElevenLabsTtsPlayer.kt` sadece Android SDK ve Kotlin Coroutines kullanir.
`build.gradle`'a eklenmesi gereken tek sey (muhtemelen zaten vardir):

```gradle
implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
```

---

## Notlar
- Her yeni kelime geldiginde onceki istek iptal edilir, yeni kelime baslar
- Hata loglarini `Logcat -> ElevenLabsTts` tag'i ile takip edebilirsin
- API key ve Voice ID dosyanin icinde hardcoded — production'da BuildConfig'e tasiyabilirsin
