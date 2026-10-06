package com.speechfree.tts

import android.content.Context
import android.media.MediaPlayer
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * SpeechFree — ElevenLabs Gercek Zamanli TTS
 *
 * Kullanim (Activity / Fragment icinde):
 *
 *   private val tts = ElevenLabsTtsPlayer(context, apiKey, voiceId)
 *
 *   // Mert'in classifier ciktisi geldiginde:
 *   tts.speak(wordIndex)
 *
 *   // Activity onDestroy:
 *   tts.release()
 *
 * Gecikme tahmini: ~550ms - 1100ms (ag + uretim + MediaPlayer)
 */
class ElevenLabsTtsPlayer(
    private val context: Context,
    // Inject from BuildConfig / secure storage. Never commit a key; for production put the call
    // behind your own backend instead of shipping an API key inside the APK.
    private val apiKey: String,
    private val voiceId: String,
) {

    private val model   = "eleven_multilingual_v2"

    private val scope = CoroutineScope(Dispatchers.IO)
    private var currentJob: Job? = null
    private var player: MediaPlayer? = null

    // Mert'in classifier index sirasi — classifier ile birebir eslestirilmeli
    private val words = mapOf(
        0 to "Evet",
        1 to "Hayir",
        2 to "Merhaba",
        3 to "Tessekurler",
        4 to "Su",
        5 to "Yardim",
        6 to "Tamam",
        7 to "Dur",
        8 to "Gel",
        9 to "Gunaydin",
    )

    /** Classifier word_index (0-9) ile ses uret ve oynat */
    fun speak(wordIndex: Int) {
        val text = words[wordIndex]
        if (text == null) {
            Log.w(TAG, "Bilinmeyen index: $wordIndex")
            return
        }
        synthesizeAndPlay(text)
    }

    private fun synthesizeAndPlay(text: String) {
        // Onceki istek iptal et, yeni kelime baslasin
        currentJob?.cancel()
        currentJob = scope.launch {
            val startMs = System.currentTimeMillis()
            Log.d(TAG, "Istek basliyor: $text")

            val audioBytes = callElevenLabs(text) ?: run {
                Log.e(TAG, "API hatasi: $text")
                return@launch
            }

            val elapsed = System.currentTimeMillis() - startMs
            Log.d(TAG, "API yaniti: ${elapsed}ms, ${audioBytes.size / 1024}KB")

            val tmpFile = File(context.cacheDir, "tts_output.mp3")
            tmpFile.writeBytes(audioBytes)

            withContext(Dispatchers.Main) {
                playFile(tmpFile)
            }
        }
    }

    private fun callElevenLabs(text: String): ByteArray? {
        return try {
            val url = URL("https://api.elevenlabs.io/v1/text-to-speech/$voiceId")
            val body = """
                {
                  "text": "$text",
                  "model_id": "$model",
                  "output_format": "mp3_44100_128",
                  "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.85
                  }
                }
            """.trimIndent()

            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                setRequestProperty("xi-api-key", apiKey)
                setRequestProperty("Content-Type", "application/json")
                connectTimeout = 5000
                readTimeout = 10000
                doOutput = true
                outputStream.write(body.toByteArray())
            }

            if (conn.responseCode != 200) {
                Log.e(TAG, "HTTP ${conn.responseCode}: ${conn.errorStream?.bufferedReader()?.readText()}")
                return null
            }

            conn.inputStream.readBytes()
        } catch (e: Exception) {
            Log.e(TAG, "Ag hatasi", e)
            null
        }
    }

    private fun playFile(file: File) {
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(file.absolutePath)
            prepare()
            start()
            setOnCompletionListener { release() }
        }
    }

    fun release() {
        currentJob?.cancel()
        player?.release()
        player = null
    }

    companion object {
        private const val TAG = "ElevenLabsTts"
    }
}
