# VAD Comparison Report

**threshold_k = 1.5**  |  Calibration = 13 s  |  Recordings: kayit1–kayit4  |  IoU match threshold = 0.3

## Overall metrics

| Metric | Value |
|---|---|
| match_rate | 66.0% (66/100) |
| false_positive_count (total) | 69 |
| miss_count (total) | 34 |
| Δonset mean ± std | 622 ± 639 ms |
| Δonset MAE | 710 ms |
| mean IoU (matched pairs) | 0.556 |
| low_snr segments (<3.0 dB) | 66 |

## Per-recording

| Recording | match_rate | FP | Miss | Δonset MAE (ms) | mean IoU |
|---|---|---|---|---|---|
| kayit1 | 66.0% | 69 | 34 | 710 | 0.556 |

## Threshold selection

Used `threshold_k=1.5` (VAD_CONFIG default = 3.0).  Lower k (e.g. 2.5) increases sensitivity but may raise FP count in recordings where PZT bleeds into EMG.  Re-run with `--threshold_k 2.5` to compare.

## Observed failure modes

- **Merged reps**: fast consecutive repetitions where inter-rep silence < `silence_gap_samples` (200 ms) may be collapsed into a single VAD segment → miss.
- **False positives**: non-speech muscle artifacts or PZT mechanical noise exceeding threshold → spurious VAD segments.
- **Low-SNR segments**: 66 matched segments below 3.0 dB flagged `low_snr=True` — classifier likely mis-classifies these.

## Accuracy comparison

| Setup | Within-session Top-1 | Top-3 | Cross-session Top-1 |
|---|---|---|---|
| Previous baseline | 67% | 87% | 52% |
| New VAD segments  | _run train_base_model.py_ | — | — |

_After running `build_dataset_recordings.py` → `train_base_model.py`, fill in results above._
