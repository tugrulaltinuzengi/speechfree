"""
06_deliverable/tts_assets klasorundeki 10 kelimeyi ElevenLabs ile yeniden uret.
Kullanim:
    python gen_tts_assets.py
"""

import os
import sys
from pathlib import Path
from elevenlabs.client import ElevenLabs

API_KEY  = os.environ["ELEVENLABS_API_KEY"]   # never hard-code; export it in your shell
VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]  # your own cloned/stock voice id
MODEL    = "eleven_multilingual_v2"

# Turkce karakterlerle dogru telaffuz
WORDS = {
    "EVET":        "Evet",
    "HAYIR":       "Hayır",
    "MERHABA":     "Merhaba",
    "TESEKKURLER": "Teşekkürler",
    "SU":          "Su",
    "YARDIM":      "Yardım",
    "TAMAM":       "Tamam",
    "DUR":         "Dur",
    "GEL":         "Gel",
    "GUNAYDIN":    "Günaydın",
}

VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.85,
}

def main():
    out_dir = Path(__file__).parent / "tts_assets"
    out_dir.mkdir(exist_ok=True)

    client = ElevenLabs(api_key=API_KEY)

    # Mevcut modelleri listele
    try:
        models = client.models.get_all()
        print("Mevcut modeller:")
        for m in models:
            print(f"  {m.model_id}  ({m.name})")
        print()
    except Exception as e:
        print(f"Model listesi alinamadi: {e}\n")

    print(f"Kullanilan model : {MODEL}")
    print(f"Voice ID         : {VOICE_ID}")
    print(f"Cikti klasoru    : {out_dir}\n")

    ok, fail = 0, 0
    for fname, text in WORDS.items():
        out_path = out_dir / f"{fname}.mp3"
        print(f"  [{fname}] '{text}' ...", end=" ", flush=True)
        try:
            audio_gen = client.text_to_speech.convert(
                voice_id=VOICE_ID,
                text=text,
                model_id=MODEL,
                output_format="mp3_44100_128",
                voice_settings=VOICE_SETTINGS,
            )
            audio_bytes = b"".join(audio_gen)
            out_path.write_bytes(audio_bytes)
            kb = len(audio_bytes) / 1024
            print(f"OK  {kb:.1f} KB")
            ok += 1
        except Exception as e:
            print(f"HATA: {e}")
            fail += 1

    print(f"\n{ok}/10 tamamlandi, {fail} hata.")
    if fail:
        sys.exit(1)

if __name__ == "__main__":
    main()
