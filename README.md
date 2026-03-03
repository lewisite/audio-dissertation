# Codec A: Explainable Neural Audio Codec
**Dissertation: "The Effect of Explainability on End-User Trust in Neural Audio Codecs"**

## Research Goal
Build an explainable neural audio codec (Codec A) on top of Meta's EnCodec that not only
compresses audio efficiently but provides end-users with transparent, interpretable feedback
about what the codec is doing — enabling calibrated trust rather than blind reliance.

## Research Questions
- **RQ1:** Does explainability increase end-user trust in a high-stakes audio communication scenario?
- **RQ2:** Does adding explainability degrade audio quality or latency compared to a conventional codec?

---

## Project Structure

```
Dissertation Code/
├── src/
│   ├── pipeline.py            # Core Codec A explainability engine (run this first)
│   └── compare_bandwidths.py  # Multi-bandwidth quality comparison
├── data/
│   ├── raw_audio/             # Original input audio files (store here)
│   ├── processed/             # Pipeline outputs (auto-created per run)
│   └── residuals/             # Standalone residual files (optional)
├── figures/                   # Final dissertation figures (curated copies)
├── experiments/               # Experiment configs and logs
├── notes/                     # Analysis notes (analysis_notes.md goes here)
├── encodec-test.py            # Original sanity-check script
├── generate-wav.py            # Generates test.wav (440 Hz sine wave)
└── experiment_log.md          # Running log of all experiments
```

---

## Quick Start

**1. Generate a test audio file (if you don't have one):**
```bash
python generate-wav.py
```

**2. Run the full Codec A explainability pipeline:**
```bash
python src/pipeline.py --input test.wav --bandwidth 6.0
```

**3. Compare across all bandwidths (1.5 / 3 / 6 / 12 / 24 kbps):**
```bash
python src/compare_bandwidths.py --input test.wav
```

**4. Use your own audio:**
```bash
python src/pipeline.py --input data/raw_audio/your_file.wav --bandwidth 6.0
```

---

## Pipeline Outputs (per run)

Each run creates a timestamped folder in `data/processed/`, containing:

| File | Contents |
|---|---|
| `reconstructed.wav` | Decoded audio (what the user hears) |
| `residual.wav` | What the codec discarded — play this to hear losses |
| `metrics.json` | All computed metrics (feeds the dashboard) |
| `summary.txt` | Plain-language explainability summary (Codec A feature) |
| `fig_waveforms.png` | Original / Reconstructed / Residual waveform comparison |
| `fig_spectrograms.png` | Spectrogram comparison (original vs reconstructed) |
| `fig_codebook_heatmap.png` | Token IDs across codebook layers and time |
| `fig_codebook_stats.png` | Entropy / usage / temporal change rate per codebook |

---

## Metrics Computed

### Waveform (time-domain)
| Metric | What it measures |
|---|---|
| `snr_db` | Signal-to-noise ratio — overall preservation quality |
| `si_sdr_db` | Scale-invariant SDR — robust distortion measure |
| `psnr_db` | Peak SNR |
| `mse` | Mean squared error between original and reconstructed |
| `residual_energy` | Energy of what was discarded |
| `residual_peak` | Loudest artifact in the residual |

### Spectral (frequency-domain)
| Metric | What it measures |
|---|---|
| `spectral_snr_db` | SNR computed on STFT magnitudes |
| `log_spectral_distance` | Perceptually grounded spectral fidelity (lower = better) |
| `spectral_mse` | MSE between STFT magnitude maps |

### Token / Codebook (interpretability core)
| Metric | What it measures |
|---|---|
| `entropy_bits` | Diversity of token choices per codebook (higher = richer info) |
| `usage_rate` | Fraction of the 1024-token vocabulary actually used |
| `temporal_change_rate` | How dynamically the codec is switching tokens frame-to-frame |
| `top_token_frequency` | How dominant the most-used token is (high = repetitive audio) |

---

## EnCodec Configuration

| Setting | Value |
|---|---|
| Model | `encodec_model_24khz` (24 kHz mono) |
| Bandwidths | 1.5 / 3.0 / 6.0 / 12.0 / 24.0 kbps |
| Codebooks (n_q) | 2 / 4 / 8 / 16 / 32 (increases with bandwidth) |
| Codebook size | 1024 tokens |
| License | CC-BY-NC 4.0 (non-commercial research) |

---

## Dependencies
- `torch` — PyTorch
- `encodec` — Meta's EnCodec
- `soundfile` — audio I/O (no ffmpeg/torchaudio required)
- `numpy`, `scipy`, `matplotlib` — analysis and visualisation
