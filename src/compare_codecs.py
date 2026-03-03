"""
3-Way Codec Comparison: Codec A vs Codec B vs Opus
====================================================
Dissertation comparison engine supporting both research questions.

Codec A  — EnCodec WITH full explainability (Codec A pipeline)
Codec B  — EnCodec WITHOUT explainability (black box, same neural model)
Opus     — Traditional codec baseline (Opus RFC 6716)

Research question mapping:
  RQ1 (trust via explainability) : Codec A vs Codec B
      Same neural model, same audio quality, only difference is the
      explainability features. Any trust difference is attributable
      to transparency alone.

  RQ2 (performance trade-off)   : Codec A vs Opus
      Neural explainable codec vs traditional state-of-the-art.
      Documents whether Codec A achieves competitive quality.

  Additional insight             : Codec B vs Opus
      Confirms that the underlying neural codec is competitive
      independent of explainability features.

Outputs (saved to data/processed/<run_id>_3way/):
  comparison.json        — all metrics for all three codecs (dashboard input)
  fig_comparison.png     — 6-panel metric comparison figure
  fig_waveforms.png      — side-by-side waveform overlays
  fig_spectrograms.png   — side-by-side spectrogram grid
  summary_table.txt      — printed comparison table

Usage:
  python src/compare_codecs.py --input test.wav
  python src/compare_codecs.py --input data/raw_audio/speech.wav --bandwidth 6.0 --opus-bitrate 12000
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (
    load_audio, save_audio,
    encode_audio, decode_audio,
    compute_waveform_metrics, compute_spectral_metrics, analyze_tokens,
)
from codec_b  import run_codec_b
from codec_opus import run_codec_opus, _encode_decode_opus
from encodec import EncodecModel


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_codec_a_metrics(input_path: str, bandwidth: float):
    """Run Codec A and return metrics dict + aligned audio arrays."""
    audio_np, src_sr = load_audio(input_path)

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model.eval()

    t0 = time.perf_counter()
    frames, wav_tensor, codes_np = encode_audio(model, audio_np, src_sr)
    encode_time = time.perf_counter() - t0

    recon_np         = decode_audio(model, frames)
    orig_at_model_sr = wav_tensor[0].cpu().numpy()
    T                = min(orig_at_model_sr.shape[-1], recon_np.shape[-1])
    orig_aligned     = orig_at_model_sr[..., :T]
    recon_aligned    = recon_np[..., :T]
    n_q, n_frames    = codes_np.shape

    wm = compute_waveform_metrics(orig_aligned, recon_aligned)
    sm = compute_spectral_metrics(orig_aligned, recon_aligned, model.sample_rate)
    tm = analyze_tokens(codes_np)

    metrics = {
        "codec":               "A",
        "label":               f"Codec A (EnCodec {bandwidth} kbps + explainability)",
        "bandwidth_kbps":      bandwidth,
        "encode_time_seconds": float(encode_time),
        "n_codebooks":         n_q,
        "n_frames":            n_frames,
        "sample_rate":         model.sample_rate,
        "waveform":            wm,
        "spectral":            sm,
        "tokens":              tm,
    }
    return metrics, orig_aligned, recon_aligned, model.sample_rate


def _run_codec_b_metrics(input_path: str, bandwidth: float):
    """Run Codec B and return metrics dict + aligned audio arrays."""
    audio_np, src_sr = load_audio(input_path)

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model.eval()

    t0 = time.perf_counter()
    frames, wav_tensor, codes_np = encode_audio(model, audio_np, src_sr)
    encode_time = time.perf_counter() - t0

    recon_np         = decode_audio(model, frames)
    orig_at_model_sr = wav_tensor[0].cpu().numpy()
    T                = min(orig_at_model_sr.shape[-1], recon_np.shape[-1])
    orig_aligned     = orig_at_model_sr[..., :T]
    recon_aligned    = recon_np[..., :T]
    n_q, n_frames    = codes_np.shape

    wm = compute_waveform_metrics(orig_aligned, recon_aligned)
    sm = compute_spectral_metrics(orig_aligned, recon_aligned, model.sample_rate)
    tm = analyze_tokens(codes_np)

    metrics = {
        "codec":               "B",
        "label":               f"Codec B (EnCodec {bandwidth} kbps, black box)",
        "bandwidth_kbps":      bandwidth,
        "encode_time_seconds": float(encode_time),
        "n_codebooks":         n_q,
        "n_frames":            n_frames,
        "sample_rate":         model.sample_rate,
        "waveform":            wm,
        "spectral":            sm,
        "tokens":              tm,
    }
    return metrics, orig_aligned, recon_aligned, model.sample_rate


def _run_opus_metrics(input_path: str, bitrate_bps: int):
    """Run Opus and return metrics dict + aligned audio arrays."""
    audio_np, src_sr = load_audio(input_path)
    if audio_np.shape[0] > 1:
        audio_mono = audio_np.mean(axis=0)
    else:
        audio_mono = audio_np[0]

    t0 = time.perf_counter()
    recon_mono = _encode_decode_opus(audio_mono, src_sr, bitrate_bps)
    encode_time = time.perf_counter() - t0

    T = min(len(audio_mono), len(recon_mono))
    orig_aligned  = audio_mono[:T].reshape(1, -1)
    recon_aligned = recon_mono[:T].reshape(1, -1)

    wm = compute_waveform_metrics(orig_aligned, recon_aligned)
    sm = compute_spectral_metrics(orig_aligned, recon_aligned, src_sr)

    metrics = {
        "codec":               "Opus",
        "label":               f"Opus ({bitrate_bps/1000:.0f} kbps, traditional)",
        "bitrate_kbps":        bitrate_bps / 1000,
        "encode_time_seconds": float(encode_time),
        "sample_rate":         src_sr,
        "waveform":            wm,
        "spectral":            sm,
    }
    return metrics, orig_aligned, recon_aligned, src_sr


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_metric_comparison(results: list, out_path: str, audio_name: str):
    """
    6-panel bar chart comparing key metrics across all three codecs.
    Designed to be dissertation-ready (clear labels, no chart junk).
    """
    labels  = [r["label"] for r in results]
    colors  = ["steelblue", "darkorange", "crimson"]

    metrics_to_plot = [
        ("waveform.snr_db",           "SNR (dB)",
         "Waveform SNR\n(higher = better reconstruction)"),
        ("waveform.si_sdr_db",        "SI-SDR (dB)",
         "Scale-Invariant SDR\n(higher = better)"),
        ("spectral.spectral_snr_db",  "Spectral SNR (dB)",
         "Spectral SNR\n(higher = better tonal fidelity)"),
        ("spectral.log_spectral_distance", "Log Spectral Distance",
         "Log Spectral Distance\n(lower = better perceptual match)"),
        ("waveform.residual_energy",  "Residual Energy",
         "Residual Energy\n(lower = less discarded)"),
        ("encode_time_seconds",       "Encode Time (s)",
         "Encode+Decode Time\n(lower = faster)"),
    ]

    def _get(d, key):
        parts = key.split(".")
        for p in parts:
            d = d.get(p, {})
        return d if isinstance(d, (int, float)) else 0.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"3-Way Codec Comparison — {audio_name}\n"
        f"Codec A (EnCodec + explainability)  |  Codec B (EnCodec black box)  |  Opus (traditional)",
        fontsize=12, fontweight="bold"
    )

    for ax, (key, ylabel, title), color in zip(
        axes.flat, metrics_to_plot,
        ["steelblue", "darkorange", "seagreen", "crimson", "purple", "saddlebrown"]
    ):
        values = [_get(r, key) for r in results]
        short_labels = [r["codec"] for r in results]
        bars = ax.bar(short_labels, values,
                      color=["steelblue", "darkorange", "crimson"],
                      edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bar values
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def _plot_waveform_comparison(orig, recon_a, recon_b, recon_opus, sr, out_path):
    """Four-panel waveform overlay: original + each codec's reconstruction."""
    T = min(orig.shape[-1], recon_a.shape[-1], recon_b.shape[-1], recon_opus.shape[-1])
    t = np.arange(T) / sr

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Waveform Comparison — Original vs Each Codec", fontsize=13, fontweight="bold")

    panels = [
        (orig[0, :T],       "Original (uncompressed)", "black"),
        (recon_a[0, :T],    "Codec A (EnCodec + explainability)", "steelblue"),
        (recon_b[0, :T],    "Codec B (EnCodec black box)",        "darkorange"),
        (recon_opus[0, :T], "Opus (traditional)",                 "crimson"),
    ]
    for ax, (sig, label, color) in zip(axes, panels):
        ax.plot(t, sig, color=color, linewidth=0.6, alpha=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def _plot_spectrogram_grid(orig, recon_a, recon_b, recon_opus, sr, out_path):
    """2×2 spectrogram grid: original / Codec A / Codec B / Opus."""
    T = min(orig.shape[-1], recon_a.shape[-1], recon_b.shape[-1], recon_opus.shape[-1])
    nperseg = min(512, max(32, T // 8))

    def _mag_db(sig):
        _, _, Z = stft(sig.astype(np.float64), fs=sr, nperseg=nperseg)
        return 20 * np.log10(np.abs(Z) + 1e-8)

    mags = [
        _mag_db(orig[0, :T]),
        _mag_db(recon_a[0, :T]),
        _mag_db(recon_b[0, :T]),
        _mag_db(recon_opus[0, :T]),
    ]
    titles = [
        "Original", "Codec A (EnCodec + explainability)",
        "Codec B (EnCodec black box)", "Opus (traditional)",
    ]
    vmin = min(m.min() for m in mags)
    vmax = max(m.max() for m in mags)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Spectrogram Comparison (dB)", fontsize=13, fontweight="bold")

    for ax, mag, title in zip(axes.flat, mags, titles):
        im = ax.imshow(mag, aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Frequency bin")

    plt.colorbar(im, ax=axes, label="Magnitude (dB)", shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def _print_table(results: list, duration: float):
    """Print a formatted comparison table to stdout."""
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  3-WAY CODEC COMPARISON TABLE")
    print(sep)
    print(f"  {'Metric':<30} {'Codec A':>12} {'Codec B':>12} {'Opus':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    def _row(label, key, fmt=".2f"):
        vals = []
        for r in results:
            parts = key.split(".")
            v = r
            for p in parts:
                v = v.get(p, {})
            vals.append(v if isinstance(v, (int, float)) else float("nan"))
        print(f"  {label:<30} {vals[0]:>{12+len(fmt)-2}{fmt}} "
              f"{vals[1]:>{12+len(fmt)-2}{fmt}} {vals[2]:>{12+len(fmt)-2}{fmt}}")

    _row("SNR (dB)",               "waveform.snr_db")
    _row("SI-SDR (dB)",            "waveform.si_sdr_db")
    _row("PSNR (dB)",              "waveform.psnr_db")
    _row("MSE",                    "waveform.mse",           ".6f")
    _row("Residual energy",        "waveform.residual_energy", ".6f")
    _row("Residual peak",          "waveform.residual_peak")
    _row("Spectral SNR (dB)",      "spectral.spectral_snr_db")
    _row("Log Spectral Distance",  "spectral.log_spectral_distance")
    _row("Encode time (s)",        "encode_time_seconds")

    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    # Bitrate / bandwidth row
    bw_a    = results[0].get("bandwidth_kbps", "—")
    bw_b    = results[1].get("bandwidth_kbps", "—")
    bw_opus = results[2].get("bitrate_kbps",   "—")
    print(f"  {'Bitrate (kbps)':<30} {str(bw_a):>12} {str(bw_b):>12} {str(bw_opus):>12}")
    print(f"  {'Duration (s)':<30} {duration:>12.2f} {duration:>12.2f} {duration:>12.2f}")

    print(sep)
    print()
    print("  RQ1 SIGNAL (Codec A vs B — same neural model, explainability differs):")
    snr_a = results[0]["waveform"]["snr_db"]
    snr_b = results[1]["waveform"]["snr_db"]
    print(f"    SNR difference Codec A - Codec B : {snr_a - snr_b:+.2f} dB")
    print(f"    (near zero expected -- trust difference is due to transparency, not audio quality)")
    print()
    print("  RQ2 SIGNAL (Codec A vs Opus -- neural+explainable vs traditional):")
    snr_o = results[2]["waveform"]["snr_db"]
    print(f"    SNR difference Codec A - Opus    : {snr_a - snr_o:+.2f} dB")
    print(f"    Encode time Codec A / Opus        : "
          f"{results[0]['encode_time_seconds']:.3f}s / {results[2]['encode_time_seconds']:.3f}s")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(input_path: str, bandwidth: float = 6.0,
                   opus_bitrate_bps: int = 12000,
                   output_dir: str = "data/processed") -> tuple:
    """
    Full 3-way comparison: Codec A vs Codec B vs Opus.

    Args:
        input_path       : path to input .wav
        bandwidth        : EnCodec bandwidth for Codec A and B (kbps)
        opus_bitrate_bps : Opus bitrate in bps (default 12000 = 12 kbps)
        output_dir       : parent directory for outputs

    Returns:
        (comparison dict, output_directory Path)
    """
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_3way"
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_name = Path(input_path).stem

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  3-WAY CODEC COMPARISON")
    print(sep)
    print(f"  Input        : {input_path}")
    print(f"  Codec A/B BW : {bandwidth} kbps (EnCodec)")
    print(f"  Opus bitrate : {opus_bitrate_bps/1000:.0f} kbps")
    print(f"  Output       : {out_dir}\n")

    # ── Run all three codecs ─────────────────────────────────────────────────
    print("Running Codec A (EnCodec + explainability)...")
    metrics_a, orig_a, recon_a, sr_a = _run_codec_a_metrics(input_path, bandwidth)
    print(f"  SNR: {metrics_a['waveform']['snr_db']:.1f} dB  |  "
          f"Time: {metrics_a['encode_time_seconds']:.3f}s\n")

    print("Running Codec B (EnCodec black box)...")
    metrics_b, orig_b, recon_b, sr_b = _run_codec_b_metrics(input_path, bandwidth)
    print(f"  SNR: {metrics_b['waveform']['snr_db']:.1f} dB  |  "
          f"Time: {metrics_b['encode_time_seconds']:.3f}s\n")

    print("Running Opus (traditional baseline)...")
    metrics_o, orig_o, recon_o, sr_o = _run_opus_metrics(input_path, opus_bitrate_bps)
    print(f"  SNR: {metrics_o['waveform']['snr_db']:.1f} dB  |  "
          f"Time: {metrics_o['encode_time_seconds']:.3f}s\n")

    results = [metrics_a, metrics_b, metrics_o]

    # Audio duration (from Codec A)
    audio_np, src_sr = load_audio(input_path)
    duration = audio_np.shape[-1] / src_sr

    # ── Print table ──────────────────────────────────────────────────────────
    _print_table(results, duration)

    # ── Save comparison JSON ─────────────────────────────────────────────────
    comparison = {
        "run_id":        run_id,
        "input_file":    str(input_path),
        "duration_s":    duration,
        "codec_a_bandwidth_kbps":  bandwidth,
        "codec_b_bandwidth_kbps":  bandwidth,
        "opus_bitrate_kbps":       opus_bitrate_bps / 1000,
        "codecs":        results,
        "rq1_snr_diff_a_minus_b":  (metrics_a["waveform"]["snr_db"] -
                                    metrics_b["waveform"]["snr_db"]),
        "rq2_snr_diff_a_minus_opus":(metrics_a["waveform"]["snr_db"] -
                                    metrics_o["waveform"]["snr_db"]),
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"      Saved: {out_dir / 'comparison.json'}")

    # ── Visualisations ───────────────────────────────────────────────────────
    print("Generating visualisations...")

    # Align all audio to same length and sample rate for plots
    # Use sr_a (model SR) as reference; Opus audio is at src_sr — resample if needed
    if sr_o != sr_a:
        from scipy.signal import resample_poly
        from math import gcd
        g    = gcd(sr_a, sr_o)
        up   = sr_a // g
        down = sr_o // g
        recon_o_r = resample_poly(recon_o[0], up, down)
        orig_o_r  = resample_poly(orig_o[0],  up, down)
        recon_o   = recon_o_r.reshape(1, -1)
        orig_a_ref = orig_a   # already at sr_a
    else:
        orig_a_ref = orig_a

    T_plot = min(orig_a_ref.shape[-1], recon_a.shape[-1],
                 recon_b.shape[-1], recon_o.shape[-1])

    _plot_metric_comparison(results, str(out_dir / "fig_comparison.png"), audio_name)

    _plot_waveform_comparison(
        orig_a_ref[..., :T_plot],
        recon_a[..., :T_plot],
        recon_b[..., :T_plot],
        recon_o[..., :T_plot],
        sr_a,
        str(out_dir / "fig_waveforms.png"),
    )

    _plot_spectrogram_grid(
        orig_a_ref[..., :T_plot],
        recon_a[..., :T_plot],
        recon_b[..., :T_plot],
        recon_o[..., :T_plot],
        sr_a,
        str(out_dir / "fig_spectrograms.png"),
    )

    print(f"\n  All outputs saved to: {out_dir}")
    print(sep + "\n")
    return comparison, out_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-way codec comparison: Codec A vs Codec B vs Opus"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input audio file (.wav)"
    )
    parser.add_argument(
        "--bandwidth", type=float, default=6.0,
        choices=[1.5, 3.0, 6.0, 12.0, 24.0],
        help="EnCodec bandwidth for Codec A and B in kbps (default: 6.0)"
    )
    parser.add_argument(
        "--opus-bitrate", type=int, default=12000,
        help="Opus target bitrate in bps (default: 12000 = 12 kbps)"
    )
    parser.add_argument(
        "--output-dir", default="data/processed",
        help="Parent directory for outputs (default: data/processed)"
    )
    args = parser.parse_args()
    run_comparison(args.input, args.bandwidth, args.opus_bitrate, args.output_dir)
