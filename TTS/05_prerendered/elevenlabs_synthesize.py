"""
ElevenLabs SDK ile 10 hedef kelimeyi sentezle -> MP3 olarak kaydet
"""

import os
from pathlib import Path
from elevenlabs.client import ElevenLabs

API_KEY  = os.environ["ELEVENLABS_API_KEY"]   # never hard-code; export it in your shell
VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]  # your own cloned/stock voice id
MODEL    = "eleven_multilingual_v2"

WORDS = {
    "EVET":        "Evet",
    "HAYIR":       "Hayir",
    "MERHABA":     "Merhaba",
    "TESEKKURLER": "Tessekurler",
    "SU":          "Su",
    "YARDIM":      "Yardim",
    "TAMAM":       "Tamam",
    "DUR":         "Dur",
    "GEL":         "Gel",
    "GUNAYDIN":    "Gunaydin",
}

def main():
    out_dir = Path(__file__).parent / "prerendered"
    out_dir.mkdir(exist_ok=True)

    client = ElevenLabs(api_key=API_KEY)

    print("Sentezleniyor...\n")

    for fname, text in WORDS.items():
        out_path = out_dir / f"{fname}.mp3"
        print(f"  {text} ...", end=" ", flush=True)

        audio_gen = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id=MODEL,
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.85,
            },
        )

        audio_bytes = b"".join(audio_gen)
        out_path.write_bytes(audio_bytes)

        kb = len(audio_bytes) // 1024
        print(f"OK ({kb} KB) -> {out_path.name}")

    mp3s = list(out_dir.glob("*.mp3"))
    print(f"\nTamam: {len(mp3s)}/10 MP3 -> {out_dir}")

if __name__ == "__main__":
    main()
