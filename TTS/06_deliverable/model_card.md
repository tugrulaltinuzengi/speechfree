# Model Card — SpeachFree TTS (Turkish, Single Speaker)

## Overview
| Field | Value |
|-------|-------|
| Model architecture | VITS2 (end-to-end, vocoder included) |
| Language | Turkish (tr-TR) |
| Task | Single-speaker text-to-speech |
| Export format | ONNX (encoder + vocoder separate) |
| Target deployment | Android via ONNX Runtime |

## Speaker
| Field | Value |
|-------|-------|
| Speaker ID | S01 |
| Gender | — (fill in) |
| Native language | Turkish |
| Recording role | — Patient / Stand-in speaker (fill in) |

## Training Data
| Field | Value |
|-------|-------|
| Total hours | — (fill in after recording) |
| Number of utterances | — |
| Recording environment | — (fill in: room type, mic model) |
| Fonetik diversity | ~300 phonetically balanced sentences + 10 target words × 20–30 takes |
| Format | 22050 Hz, mono, 16-bit WAV |

## Pretrained Baseline
| Field | Value |
|-------|-------|
| Starting model | — (fill in: e.g. coqui tts_models/tr/common-voice/glow-tts or VCTK VITS) |
| Fine-tune epochs | — |
| Hardware | — (fill in: GPU model, training time) |

## Evaluation
| Metric | Value | Target |
|--------|-------|--------|
| MCD (Mel Cepstral Distortion) | — dB | < 7 dB |
| Speaker similarity (Resemblyzer cosine) | — | > 0.85 |
| MOS (subjective, N≥5 raters) | — / 5.0 | ≥ 3.5 |
| Latency — 1 word, Android CPU | — ms | < 500 ms |
| ONNX total model size | — MB | < 30 MB |

## 10 Target Words
EVET, HAYIR, MERHABA, TEŞEKKÜRLER, SU, YARDIM, TAMAM, DUR, GEL, GÜNAYDIN

## Files in This Package
| File | Description |
|------|-------------|
| `vits_encoder.onnx` | Text encoder + duration predictor + flow |
| `vits_vocoder.onnx` | HiFi-GAN vocoder (latent → waveform) |
| `tokenizer_config.json` | Character vocabulary and text cleaner info |
| `audio_config.json` | Sample rate and mel-spectrogram parameters |
| `speaker_embedding.npy` | Fixed speaker embedding (single speaker) |
| `usage_example.py` | Python demo — mirrors Android ONNX Runtime calls |

## Known Limitations
- Vocabulary is fixed to the training script's character set — out-of-vocabulary chars map to `<UNK>`
- No real-time streaming — synthesis is per-utterance
- Turkish-specific characters (ç, ş, ğ, ı, ö, ü) require NFC Unicode normalization before tokenization

## Contact
- TTS module owner: Ahmet Tuğrul Altınüzengi
- Android integration: Yiğit
- ML classifier integration: Mert
