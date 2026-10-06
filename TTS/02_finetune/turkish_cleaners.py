"""
02_finetune/turkish_cleaners.py

Turkish text normalization for TTS training.
Drop-in replacement for coqui-tts cleaner pipeline.

Covers:
  - Number → words (Türkçe)
  - Abbreviation expansion
  - Punctuation normalization
  - Lowercase with correct Turkish i/İ handling
  - Whitespace cleanup
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Turkish locale-aware lowercase
# ---------------------------------------------------------------------------
_TR_LOWER = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                           "abcçdefgğhıijklmnoöprsştuüvyz")


def turkish_lower(text: str) -> str:
    return text.translate(_TR_LOWER)


# ---------------------------------------------------------------------------
# Number → Turkish word
# ---------------------------------------------------------------------------
_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def _three_digits(n: int) -> str:
    assert 0 <= n < 1000
    if n == 0:
        return ""
    result = ""
    hundreds = n // 100
    remainder = n % 100
    tens = remainder // 10
    ones = remainder % 10
    if hundreds:
        result += ("" if hundreds == 1 else _ONES[hundreds] + " ") + "yüz "
    if tens:
        result += _TENS[tens] + " "
    if ones:
        result += _ONES[ones] + " "
    return result.strip()


def number_to_turkish(n: int) -> str:
    if n == 0:
        return "sıfır"
    negative = n < 0
    n = abs(n)
    groups = []
    scales = ["", "bin", "milyon", "milyar", "trilyon"]
    idx = 0
    while n > 0:
        group = n % 1000
        if group:
            g_str = _three_digits(group)
            if idx == 1 and group == 1:
                g_str = ""  # "bir bin" → "bin"
            groups.append((g_str + " " + scales[idx]).strip())
        n //= 1000
        idx += 1
    result = " ".join(reversed(groups))
    if negative:
        result = "eksi " + result
    return result.strip()


def _replace_number(match) -> str:
    token = match.group(0).replace(".", "").replace(",", ".")
    try:
        if "." in token:
            int_part, dec_part = token.split(".", 1)
            word = number_to_turkish(int(int_part)) + " nokta " + " ".join(
                number_to_turkish(int(d)) for d in dec_part if d.isdigit()
            )
        else:
            word = number_to_turkish(int(token))
    except (ValueError, OverflowError):
        word = token
    return word


# ---------------------------------------------------------------------------
# Abbreviation table
# ---------------------------------------------------------------------------
_ABBREVIATIONS = {
    r"\bdr\b": "doktor",
    r"\bprof\b": "profesör",
    r"\bdoç\b": "doçent",
    r"\byr\.doç\b": "yardımcı doçent",
    r"\bapt\b": "apartman",
    r"\bblv\b": "bulvar",
    r"\bcad\b": "cadde",
    r"\bsk\b": "sokak",
    r"\bno\b": "numara",
    r"\bvb\b": "ve benzeri",
    r"\bvs\b": "ve saire",
    r"\bvd\b": "ve diğerleri",
    r"\bsn\b": "saniye",
    r"\bdak\b": "dakika",
    r"\bsaat\b": "saat",
    r"\bkg\b": "kilogram",
    r"\bgr\b": "gram",
    r"\bkm\b": "kilometre",
    r"\bm\b": "metre",
    r"\bcm\b": "santimetre",
    r"\bmm\b": "milimetre",
    r"\btl\b": "türk lirası",
    r"\busd\b": "amerikan doları",
    r"\beur\b": "euro",
}


def expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Punctuation / symbol normalization
# ---------------------------------------------------------------------------
_PUNCT_MAP = str.maketrans({
    "–": "-",
    "—": "-",
    "…": "...",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "°": " derece",
    "%": " yüzde",
    "&": " ve",
    "@": " at",
    "+": " artı",
    "=": " eşittir",
    "×": " çarpı",
    "÷": " bölü",
})


def normalize_punctuation(text: str) -> str:
    return text.translate(_PUNCT_MAP)


# ---------------------------------------------------------------------------
# Main cleaner entry point (matches coqui-tts cleaner signature)
# ---------------------------------------------------------------------------
def turkish_cleaners(text: str) -> str:
    """Full pipeline — call this from TTS config as the text cleaner."""
    # Unicode NFC
    text = unicodedata.normalize("NFC", text)
    # Abbreviations before lowercasing
    text = expand_abbreviations(text)
    # Numbers → words
    text = re.sub(r"[\d.,]+", _replace_number, text)
    # Punct normalization
    text = normalize_punctuation(text)
    # Lowercase (Turkish-aware)
    text = turkish_lower(text)
    # Remove remaining non-Turkish characters (keep letters, space, hyphen, comma, period)
    text = re.sub(r"[^\w\s\-,.]", " ", text, flags=re.UNICODE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "Dr. Ahmet 1500 TL kazandı.",
        "İstanbul'da 3.5 milyon araç var.",
        "Prof. Çelik'in ofisi No. 42'de.",
        "Hız 120 km/h idi.",
        "İ ve i harfleri Türkçede farklıdır.",
    ]
    for t in tests:
        print(f"IN : {t}")
        print(f"OUT: {turkish_cleaners(t)}")
        print()
