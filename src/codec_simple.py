"""
Codec Simple: Transparent STFT-based Spectral Codec
=====================================================
A fully handcrafted codec built from basic signal processing — no neural
networks, no machine learning, no learned parameters of any kind.

Algorithm (every step is auditable):
  1. Segment audio using a short-time Fourier transform (STFT) window
  2. For each time frame, rank all frequency bins by magnitude
  3. Keep only the top-K bins (discard the rest — set to zero)
  4. Quantise the kept coefficients to a fixed bit depth
  5. Reconstruct via inverse STFT (overlap-add)

Why this matters for the dissertation:
  - Every decision is explicit and reproducible by hand
  - Provides the starkest possible contrast with EnCodec's learned RVQ tokens
  - Demonstrates what "compression" does at the most intuitive level
  - The spectral mask figure shows users *exactly* which frequencies were kept
  - Makes the case that neural codecs sacrifice transparency for quality

Compression levels map approximately to EnCodec bandwidths:
  keep_fraction = 0.05  →  ~1-3 kbps equivalent  (heavy compression)
  keep_fraction = 0.15  →  ~3-6 kbps equivalent  (medium compression)
  keep_fraction = 0.30  →  ~6-12 kbps equivalent (light compression)
  keep_fraction = 0.60  →  ~12-24 kbps equivalent (minimal compression)

Usage:
  python src/codec_simple.py --input test.wav --keep 0.15
  python src/codec_simple.py --input data/raw_audio/speech.wav --keep 0.30
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import stft, istft
from scipy.signal.windows import hann

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (
    load_audio, save_audio,
    compute_waveform_metrics, compute_spectral_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# CODEC CORE
# ─────────────────────────────────────────────────────────────────────────────

def _quantise(values: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Uniform mid-tread quantisation to n_bits levels.
    Maps values to integers then back to float — simulates finite precision storage.
    """
    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return values.copy()
    n_levels = 2 ** n_bits - 1
    normalised = (values - v_min) / (v_max - v_min)   # 0..1
    quantised  = np.round(normalised * n_levels) / n_levels
    return quantised * (v_max - v_min) + v_min


def compress_audio(
    audio_mono: np.ndarray,
    sr: int,
    keep_fraction: float = 0.15,
    quant_bits: int = 16,
    nperseg: int = 512,
) -> tuple:
    """
    STFT spectral compression.

    For every time frame:
      - Rank all frequency bins by magnitude
      - Keep the top (keep_fraction × 100)% of bins
      - Zero out all other bins
      - Quantise kept magnitudes to quant_bits

    Args:
        audio_mono    : 1-D float32 waveform, range [-1, 1]
        sr            : sample rate in Hz
        keep_fraction : fraction of frequency bins to keep (0.0–1.0)
        quant_bits    : bit depth for quantising kept coefficients
        nperseg       : STFT window length in samples

    Returns:
        reconstructed : float32 waveform, same shape as input
        mask          : bool array [n_freqs, n_frames] — True = kept
        stft_orig     : complex STFT of original (for visualisation)
        stft_comp     : complex STFT after masking (for visualisation)
        f_axis        : frequency axis in Hz
        t_axis        : time axis in seconds
    """
    noverlap = nperseg // 2

    f_axis, t_axis, Z = stft(
        audio_mono.astype(np.float64),
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=True,
    )

    magnitude = np.abs(Z)    # [n_freqs, n_frames]
    phase     = np.angle(Z)

    n_freqs, n_frames = magnitude.shape
    n_keep = max(1, int(n_freqs * keep_fraction))

    # ── Per-frame top-K selection ─────────────────────────────────────────────
    mask = np.zeros((n_freqs, n_frames), dtype=bool)
    for t in range(n_frames):
        top_k = np.argpartition(magnitude[:, t], -n_keep)[-n_keep:]
        mask[top_k, t] = True

    # ── Apply mask and quantise kept magnitudes ───────────────────────────────
    kept_mag = magnitude * mask.astype(np.float64)

    if quant_bits < 32:
        # Quantise only the non-zero (kept) entries
        nonzero = kept_mag[mask]
        if len(nonzero) > 0:
            kept_mag[mask] = _quantise(nonzero, quant_bits)

    Z_compressed = kept_mag * np.exp(1j * phase)

    # ── Reconstruct via inverse STFT ─────────────────────────────────────────
    _, reconstructed = istft(
        Z_compressed,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Trim/pad to original length
    T = len(audio_mono)
    if len(reconstructed) > T:
        reconstructed = reconstructed[:T]
    elif len(reconstructed) < T:
        reconstructed = np.pad(reconstructed, (0, T - len(reconstructed)))

    return (
        reconstructed.astype(np.float32),
        mask,
        Z,
        Z_compressed,
        f_axis,
        t_axis,
    )


def estimate_bitrate(mask: np.ndarray, quant_bits: int, duration_s: float) -> float:
    """
    Estimate effective bitrate in kbps based on how many coefficients were kept.
    Each kept coefficient stores: magnitude (quant_bits) + phase (quant_bits) + index (10 bits).
    """
    n_kept_total = int(mask.sum())
    bits_per_coeff = quant_bits * 2 + 10    # magnitude + phase + bin index
    total_bits = n_kept_total * bits_per_coeff
    return (total_bits / duration_s) / 1000.0   # kbps


# ─────────────────────────────────────────────────────────────────────────────
# PLAIN-LANGUAGE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(
    wm: dict, sm: dict, mask: np.ndarray,
    keep_fraction: float, quant_bits: int,
    est_bitrate: float, duration: float,
) -> str:
    """
    Plain-language explainability summary for the simple codec.
    Because the algorithm is deterministic, the explanation is exact —
    not an approximation or post-hoc rationalisation.
    """
    snr       = wm["snr_db"]
    si_sdr    = wm["si_sdr_db"]
    spec_snr  = sm["spectral_snr_db"]
    lsd       = sm["log_spectral_distance"]
    n_freqs, n_frames = mask.shape
    total_bins = n_freqs * n_frames
    kept_bins  = int(mask.sum())
    disc_bins  = total_bins - kept_bins
    disc_pct   = disc_bins / total_bins * 100

    lines = [
        "=" * 62,
        "  CODEC SIMPLE  |  TRANSPARENT SPECTRAL CODEC",
        "=" * 62,
        f"  Algorithm     : STFT top-K spectral selection",
        f"  Keep fraction : {keep_fraction:.0%} of frequency bins per frame",
        f"  Quantisation  : {quant_bits}-bit uniform",
        f"  Est. bitrate  : {est_bitrate:.1f} kbps",
        f"  Duration      : {duration:.2f}s",
        "",
        "[ WHAT THIS CODEC DID (exact, not an approximation) ]",
        f"  Total frequency-time bins : {total_bins:,}",
        f"  Bins kept (transmitted)   : {kept_bins:,} ({kept_bins/total_bins:.0%})",
        f"  Bins discarded (zeroed)   : {disc_bins:,} ({disc_pct:.0f}%)",
        f"  -- These discarded bins are exactly what you hear in residual.wav.",
        f"  -- The spectral mask figure shows which bins were kept (bright)",
        f"     vs discarded (dark) at every moment in time.",
        "",
        "[ AUDIO PRESERVATION QUALITY ]",
    ]

    if snr >= 35:
        quality = "EXCELLENT -- Audio is virtually indistinguishable from the original."
    elif snr >= 25:
        quality = "GOOD -- Audio is well preserved with only subtle losses."
    elif snr >= 15:
        quality = "MODERATE -- Noticeable but acceptable differences from the original."
    elif snr >= 5:
        quality = "DEGRADED -- Clear differences from the original are present."
    else:
        quality = "SIGNIFICANT LOSS -- The audio has been substantially altered."

    lines += [
        f"  {quality}",
        f"  Waveform SNR  : {snr:.1f} dB",
        f"  SI-SDR        : {si_sdr:.1f} dB",
        "",
        "[ SPECTRAL FIDELITY ]",
    ]

    if spec_snr >= 20:
        spec_desc = "Frequency content is very well preserved."
    elif spec_snr >= 10:
        spec_desc = "Most frequency bands preserved; some detail may be softened."
    else:
        spec_desc = "Significant spectral reshaping occurred."

    lines += [
        f"  {spec_desc}",
        f"  Spectral SNR          : {spec_snr:.1f} dB",
        f"  Log Spectral Distance : {lsd:.3f}  (lower = better)",
        "",
        "[ WHAT WAS KEPT vs DISCARDED ]",
    ]

    if keep_fraction <= 0.10:
        kept_desc = "Very aggressive compression -- only the loudest frequencies retained."
    elif keep_fraction <= 0.25:
        kept_desc = "Heavy compression -- dominant tones preserved, overtones mostly discarded."
    elif keep_fraction <= 0.50:
        kept_desc = "Moderate compression -- main harmonic structure retained."
    else:
        kept_desc = "Light compression -- most spectral detail is preserved."

    lines += [
        f"  {kept_desc}",
        f"  Every frame independently selects the {keep_fraction:.0%} loudest frequency bins.",
        f"  Softer, higher-frequency, and transient content is most likely discarded.",
        "",
        "[ KEY DIFFERENCE FROM ENCODEC ]",
        "  This codec applies fixed rules (keep loudest bins) that you can",
        "  inspect directly. EnCodec LEARNS which patterns to encode via",
        "  training on thousands of hours of audio -- it discards information",
        "  that its neural network has learned humans are unlikely to notice,",
        "  not simply the quietest frequencies. That is why EnCodec achieves",
        "  better quality at the same bitrate, but at the cost of transparency.",
        "=" * 62,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectral_mask(mask: np.ndarray, f_axis: np.ndarray, t_axis: np.ndarray,
                       keep_fraction: float, out_path: str):
    """
    The core explainability figure for the simple codec.

    Shows the binary mask: bright = frequency bin was KEPT and transmitted;
    dark = frequency bin was DISCARDED (set to zero, not transmitted).

    This is the most intuitive possible visualisation of what compression
    does -- every decision is visible, nothing is hidden.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(
        mask.astype(np.float32),
        aspect="auto",
        origin="lower",
        extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]],
        cmap="RdYlGn",
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax.set_title(
        f"Spectral Selection Mask -- Simple Codec ({keep_fraction:.0%} of bins kept per frame)\n"
        f"GREEN = frequency kept and transmitted  |  RED = frequency discarded (zeroed out)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    # Add fraction annotation
    kept_pct = mask.mean() * 100
    ax.text(
        0.02, 0.97,
        f"Kept: {kept_pct:.1f}%  |  Discarded: {100-kept_pct:.1f}%",
        transform=ax.transAxes,
        fontsize=10, color="white", fontweight="bold",
        va="top", bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def plot_spectral_comparison(Z_orig, Z_comp, f_axis, t_axis, out_path):
    """
    Side-by-side spectrogram: original vs compressed (after masking + quantisation).
    Shows exactly where spectral energy was removed.
    """
    mag_orig = 20 * np.log10(np.abs(Z_orig) + 1e-8)
    mag_comp = 20 * np.log10(np.abs(Z_comp) + 1e-8)
    mag_diff = mag_orig - mag_comp   # positive where original had energy but codec removed it

    vmin = min(mag_orig.min(), mag_comp.min())
    vmax = max(mag_orig.max(), mag_comp.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Simple Codec -- Spectral Comparison\n"
        "Left: Original  |  Centre: After Compression  |  Right: What Was Removed",
        fontsize=12, fontweight="bold"
    )
    extent = [t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]]

    im1 = axes[0].imshow(mag_orig, aspect="auto", origin="lower",
                          extent=extent, vmin=vmin, vmax=vmax, cmap="magma")
    axes[0].set_title("Original Spectrum")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")

    axes[1].imshow(mag_comp, aspect="auto", origin="lower",
                   extent=extent, vmin=vmin, vmax=vmax, cmap="magma")
    axes[1].set_title("Compressed Spectrum")
    axes[1].set_xlabel("Time (s)")

    im3 = axes[2].imshow(mag_diff, aspect="auto", origin="lower",
                          extent=extent, cmap="hot", vmin=0)
    axes[2].set_title("Discarded Energy\n(bright = more energy removed here)")
    axes[2].set_xlabel("Time (s)")

    plt.colorbar(im1, ax=axes[:2], label="Magnitude (dB)", shrink=0.8)
    plt.colorbar(im3, ax=axes[2],  label="Energy removed (dB)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def plot_waveform_comparison(original, reconstructed, residual, sr, out_path):
    """Three-panel waveform: original / reconstructed / residual."""
    T = min(len(original), len(reconstructed), len(residual))
    t = np.arange(T) / sr

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Simple Codec -- Waveform Comparison", fontsize=13, fontweight="bold")

    for ax, sig, label, color in [
        (axes[0], original[:T],      "Original",                "black"),
        (axes[1], reconstructed[:T], "Reconstructed (Simple Codec)", "seagreen"),
        (axes[2], residual[:T],      "Residual (What Was Discarded)", "crimson"),
    ]:
        ax.plot(t, sig, color=color, linewidth=0.6, alpha=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def plot_frequency_retention(mask: np.ndarray, f_axis: np.ndarray, out_path: str):
    """
    Bar chart showing how often each frequency band was retained across all frames.
    Reveals which frequencies the codec considered most important (loudest on average).
    """
    retention_rate = mask.mean(axis=1)   # [n_freqs] — fraction of frames where each bin was kept

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(f_axis, retention_rate * 100, color="steelblue", edgecolor="none", height=f_axis[1] - f_axis[0])
    ax.axvline(mask.mean() * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"Overall keep rate ({mask.mean()*100:.0f}%)")
    ax.set_xlabel("Retention rate (% of frames)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        "Frequency Retention Rate -- Simple Codec\n"
        "(How often each frequency bin was kept across all frames)\n"
        "High = this frequency was loud and consistently retained",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_codec_simple(
    input_path: str,
    keep_fraction: float = 0.15,
    quant_bits: int = 16,
    output_dir: str = "data/processed",
) -> tuple:
    """
    Full simple codec pipeline.

    Args:
        input_path    : path to input .wav file
        keep_fraction : fraction of frequency bins to keep per frame (0.0–1.0)
        quant_bits    : quantisation bit depth for kept coefficients
        output_dir    : parent directory for outputs

    Returns:
        (metrics dict, output_directory Path)
    """
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_simple_k{int(keep_fraction*100)}"
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  CODEC SIMPLE  |  TRANSPARENT SPECTRAL CODEC")
    print(sep)
    print(f"  Input         : {input_path}")
    print(f"  Keep fraction : {keep_fraction:.0%} of spectral bins per frame")
    print(f"  Quantisation  : {quant_bits}-bit")
    print(f"  Output        : {out_dir}")
    print()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("[1/5] Loading audio...")
    audio_np, sr = load_audio(input_path)
    duration = audio_np.shape[-1] / sr
    print(f"      Duration : {duration:.2f}s  |  SR : {sr} Hz")

    # Use mono (average channels if stereo)
    if audio_np.shape[0] > 1:
        audio_mono = audio_np.mean(axis=0)
    else:
        audio_mono = audio_np[0]

    # ── 2. Compress ──────────────────────────────────────────────────────────
    print("[2/5] Compressing...")
    t0 = time.perf_counter()
    recon_mono, mask, Z_orig, Z_comp, f_axis, t_axis = compress_audio(
        audio_mono, sr, keep_fraction, quant_bits
    )
    encode_time = time.perf_counter() - t0
    est_bitrate = estimate_bitrate(mask, quant_bits, duration)
    print(f"      Encode time     : {encode_time:.3f}s")
    print(f"      Estimated bitrate: {est_bitrate:.1f} kbps")
    print(f"      Bins kept       : {mask.mean():.0%} of {mask.size:,} total")

    # Align lengths
    T = min(len(audio_mono), len(recon_mono))
    orig_aligned  = audio_mono[:T].reshape(1, -1)
    recon_aligned = recon_mono[:T].reshape(1, -1)
    residual_np   = orig_aligned - recon_aligned

    # ── 3. Save audio ────────────────────────────────────────────────────────
    save_audio(str(out_dir / "reconstructed.wav"), recon_aligned, sr)
    save_audio(str(out_dir / "residual.wav"),      residual_np,   sr)
    print(f"      Saved reconstructed.wav and residual.wav")

    # ── 4. Metrics ───────────────────────────────────────────────────────────
    print("[3/5] Computing metrics...")
    wm = compute_waveform_metrics(orig_aligned, recon_aligned)
    sm = compute_spectral_metrics(orig_aligned, recon_aligned, sr)

    metrics = {
        "run_id":              run_id,
        "codec":               "Simple",
        "description":         "Transparent STFT top-K spectral codec (no neural network)",
        "input_file":          str(input_path),
        "keep_fraction":       keep_fraction,
        "quant_bits":          quant_bits,
        "estimated_bitrate_kbps": est_bitrate,
        "src_sample_rate":     sr,
        "duration_seconds":    float(duration),
        "encode_time_seconds": float(encode_time),
        "n_freq_bins":         int(mask.shape[0]),
        "n_frames":            int(mask.shape[1]),
        "bins_kept":           int(mask.sum()),
        "bins_discarded":      int(mask.size - mask.sum()),
        "keep_rate_actual":    float(mask.mean()),
        "waveform":            wm,
        "spectral":            sm,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary = generate_summary(wm, sm, mask, keep_fraction, quant_bits, est_bitrate, duration)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary)
    print()
    print(summary)

    # ── 5. Visualisations ────────────────────────────────────────────────────
    print("[4/5] Generating visualisations...")
    plot_waveform_comparison(
        audio_mono[:T], recon_mono[:T], residual_np[0], sr,
        str(out_dir / "fig_waveforms.png")
    )
    plot_spectral_mask(
        mask, f_axis, t_axis, keep_fraction,
        str(out_dir / "fig_spectral_mask.png")
    )
    plot_spectral_comparison(
        Z_orig, Z_comp, f_axis, t_axis,
        str(out_dir / "fig_spectral_comparison.png")
    )
    plot_frequency_retention(
        mask, f_axis,
        str(out_dir / "fig_frequency_retention.png")
    )

    print(f"\n[5/5] All outputs saved to: {out_dir}")
    print(sep + "\n")
    return metrics, out_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple spectral codec -- transparent STFT-based compression"
    )
    parser.add_argument("--input",    required=True,  help="Path to input .wav file")
    parser.add_argument("--keep",     type=float, default=0.15,
                        help="Fraction of spectral bins to keep per frame (default: 0.15 = 15%%)")
    parser.add_argument("--bits",     type=int,   default=16,
                        help="Quantisation bit depth (default: 16)")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Parent directory for outputs (default: data/processed)")
    args = parser.parse_args()

    if not 0.01 <= args.keep <= 1.0:
        parser.error("--keep must be between 0.01 and 1.0")

    run_codec_simple(args.input, args.keep, args.bits, args.output_dir)
