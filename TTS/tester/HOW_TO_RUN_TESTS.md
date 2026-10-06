# How to Run the TTS Tests

## 1. Setup

```bash
# Install minimal deps (no training, no ONNX model needed):
pip install numpy soundfile pydub scipy

# Full install:
pip install -r ../requirements.txt
```

## 2. Run All Tests

```bash
# From the TTS/ root folder:
python tester/run_all_tests.py

# Verbose output:
python tester/run_all_tests.py --verbose

# Check deps only:
python tester/run_all_tests.py --deps_only
```

## 3. Run a Single Module

```bash
python tester/test_turkish_cleaners.py     # No deps — runs first
python tester/test_phonetic_sentences.py   # No deps — checks sentences file
python tester/test_metadata.py             # No deps — pure Python
python tester/test_preprocess_audio.py     # Needs: pydub, soundfile
python tester/test_evaluation.py           # Needs: soundfile, scipy
python tester/test_onnx_pipeline.py        # Needs: onnxruntime (live test skipped without model)
```

## 4. With pytest

```bash
pip install pytest
pytest tester/ -v
pytest tester/test_turkish_cleaners.py -v
pytest tester/ -v --tb=short -x     # stop on first failure
```

## 5. What Each Test Covers

| File | What it tests | Deps needed |
|------|--------------|-------------|
| `test_turkish_cleaners.py` | Text normalization, i/I→ı/İ, number→word, abbreviations | None |
| `test_phonetic_sentences.py` | Sentence count, phoneme coverage, diversity | None |
| `test_metadata.py` | metadata.csv creation, pipe format, normalization | None |
| `test_preprocess_audio.py` | WAV loading, resampling, normalization, segmentation | pydub, soundfile |
| `test_evaluation.py` | MCD calculation, MOS HTML generation | soundfile, scipy |
| `test_onnx_pipeline.py` | Token shapes, config files, mock ONNX pipeline, live inference | onnxruntime |

## 6. Expected Results (Before Training)

Without a trained ONNX model, the live ONNX test is **automatically skipped**.
All other tests should **pass** with only pure-Python deps installed.

After running `export_vits.py`, re-run `test_onnx_pipeline.py` — the live test
will activate and verify end-to-end synthesis on the 10 target words.
