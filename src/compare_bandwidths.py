"""
Bandwidth Comparison — Codec A Dissertation Tool
=================================================
Runs the same audio file through EnCodec at every bandwidth (1.5 / 3 / 6 / 12 / 24 kbps)
and produces a side-by-side comparison table + summary figure.

This directly supports:
  Phase 2 of the work plan  — "Support at least 3 bandwidths"
  Phase 3                   — "Compare errors across bandwidth"
  RQ2 of the dissertation   — performance trade-offs of the explainable codec

Usage:
  python src/compare_bandwidths.py --input test.wav
  python src/compare_bandwidths.py --input data/raw_audio/speech.wav --bandwidths 3.0 6.0 12.0
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the pipeline functions directly
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (
    load_audio, encode_audio, decode_audio,
    compute_waveform_metrics, compute_spectral_metrics, analyze_tokens,
    save_audio,
)
from encodec import EncodecModel
from encodec.utils import convert_audio


BANDWIDTHS = [1.5, 3.0, 6.0, 12.0, 24.0]


def run_comparison(input_path: str, bandwidths: list, output_dir: str = "data/processed"):
    """
    Run pipeline for each bandwidth and collect metrics.
    Saves a comparison table (JSON + printed) and a summary figure.
    """
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_comparison"
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  CODEC A  |  BANDWIDTH COMPARISON")
    print(sep)
    print(f"  Input      : {input_path}")
    print(f"  Bandwidths : {bandwidths} kbps")
    print(f"  Output     : {out_dir}")
    print()

    # Load audio once
    print("Loading audio...")
    audio_np, src_sr = load_audio(input_path)
    duration = audio_np.shape[-1] / src_sr
    print(f"  Duration : {duration:.2f}s  |  SR : {src_sr} Hz\n")

    results = []

    for bw in bandwidths:
        print(f"── Bandwidth {bw} kbps {'─' * 40}")
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bw)
        model.eval()

        import time
        t0 = time.perf_counter()
        frames, wav_tensor, codes_np = encode_audio(model, audio_np, src_sr)
        encode_time = time.perf_counter() - t0

        recon_np = decode_audio(model, frames)

        orig_at_model_sr = wav_tensor[0].cpu().numpy()
        T = min(orig_at_model_sr.shape[-1], recon_np.shape[-1])
        orig_a  = orig_at_model_sr[..., :T]
        recon_a = recon_np[..., :T]
        resid   = orig_a - recon_a

        wm = compute_waveform_metrics(orig_a, recon_a)
        sm = compute_spectral_metrics(orig_a, recon_a, model.sample_rate)
        tm = analyze_tokens(codes_np)

        n_q, n_frames = codes_np.shape
        print(f"   Codebooks : {n_q:>2}  |  Frames : {n_frames}  |  Encode : {encode_time:.3f}s")
        print(f"   SNR       : {wm['snr_db']:>7.2f} dB")
        print(f"   SI-SDR    : {wm['si_sdr_db']:>7.2f} dB")
        print(f"   Spec SNR  : {sm['spectral_snr_db']:>7.2f} dB")
        print(f"   LSD       : {sm['log_spectral_distance']:>7.4f}")
        print(f"   Mean Ent  : {tm['mean_entropy_bits']:>7.3f} bits")
        print(f"   Mean Chg  : {tm['mean_temporal_change_rate']:>7.1%}")
        print()

        # Save individual reconstructed and residual
        bw_str = str(bw).replace(".", "_")
        save_audio(str(out_dir / f"reconstructed_{bw_str}kbps.wav"), recon_a, model.sample_rate)
        save_audio(str(out_dir / f"residual_{bw_str}kbps.wav"),      resid,   model.sample_rate)

        results.append({
            "bandwidth_kbps":           bw,
            "n_codebooks":              n_q,
            "n_frames":                 n_frames,
            "encode_time_seconds":      encode_time,
            "snr_db":                   wm["snr_db"],
            "si_sdr_db":                wm["si_sdr_db"],
            "psnr_db":                  wm["psnr_db"],
            "mse":                      wm["mse"],
            "residual_energy":          wm["residual_energy"],
            "spectral_snr_db":          sm["spectral_snr_db"],
            "log_spectral_distance":    sm["log_spectral_distance"],
            "mean_entropy_bits":        tm["mean_entropy_bits"],
            "mean_temporal_change_rate":tm["mean_temporal_change_rate"],
            "mean_usage_rate":          tm["mean_usage_rate"],
        })

    # ── Print summary table ──────────────────────────────────────────────────
    print(sep)
    print("  COMPARISON TABLE")
    print(sep)
    header = f"  {'BW (kbps)':>10}  {'CBs':>4}  {'SNR (dB)':>9}  {'SI-SDR':>7}  {'Spec SNR':>9}  {'LSD':>6}  {'Entropy':>8}  {'Chg Rate':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['bandwidth_kbps']:>10}  "
            f"{r['n_codebooks']:>4}  "
            f"{r['snr_db']:>9.2f}  "
            f"{r['si_sdr_db']:>7.2f}  "
            f"{r['spectral_snr_db']:>9.2f}  "
            f"{r['log_spectral_distance']:>6.3f}  "
            f"{r['mean_entropy_bits']:>8.3f}  "
            f"{r['mean_temporal_change_rate']:>9.1%}"
        )
    print(sep)

    # Save JSON
    comparison_data = {
        "run_id":       run_id,
        "input_file":   str(input_path),
        "duration_s":   duration,
        "src_sr":       src_sr,
        "results":      results,
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n  Saved comparison.json")

    # ── Generate comparison figure ───────────────────────────────────────────
    _plot_comparison(results, str(out_dir / "fig_comparison.png"), input_path)
    print(f"  Saved fig_comparison.png")
    print(f"\n  All outputs saved to: {out_dir}")
    print(f"{sep}\n")
    return results, out_dir


def _plot_comparison(results, out_path, title_suffix=""):
    """
    Six-panel comparison figure across bandwidths.
    Each panel shows how one metric changes as bitrate increases.
    """
    bws   = [r["bandwidth_kbps"]           for r in results]
    snrs  = [r["snr_db"]                   for r in results]
    sisdrs= [r["si_sdr_db"]                for r in results]
    ssnrs = [r["spectral_snr_db"]          for r in results]
    lsds  = [r["log_spectral_distance"]    for r in results]
    ents  = [r["mean_entropy_bits"]        for r in results]
    chgs  = [r["mean_temporal_change_rate"]for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Codec A — Bandwidth vs Quality/Complexity Trade-off\n{Path(title_suffix).name}",
        fontsize=13, fontweight="bold"
    )

    panels = [
        (axes[0, 0], snrs,   "SNR (dB)",                   "steelblue",  "Waveform SNR\n(higher = better preservation)"),
        (axes[0, 1], sisdrs, "SI-SDR (dB)",                "darkorange", "Scale-Invariant SDR\n(higher = better, robust to amplitude)"),
        (axes[0, 2], ssnrs,  "Spectral SNR (dB)",          "purple",     "Frequency-Domain SNR\n(higher = better tonal fidelity)"),
        (axes[1, 0], lsds,   "Log Spectral Distance",      "crimson",    "Log Spectral Distance\n(lower = better perceptual match)"),
        (axes[1, 1], ents,   "Mean Codebook Entropy (bits)","seagreen",  "Mean Token Entropy\n(reflects audio complexity)"),
        (axes[1, 2], chgs,   "Mean Change Rate",            "saddlebrown","Mean Temporal Change Rate\n(fraction of frames with new token)"),
    ]
    for ax, values, ylabel, color, title in panels:
        ax.plot(bws, values, "o-", color=color, linewidth=2, markersize=7)
        ax.set_xlabel("Bandwidth (kbps)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(bws)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Codec A: Compare EnCodec performance across bandwidths"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input audio file (.wav)"
    )
    parser.add_argument(
        "--bandwidths", nargs="+", type=float, default=BANDWIDTHS,
        help="Bandwidths to compare (default: 1.5 3.0 6.0 12.0 24.0)"
    )
    parser.add_argument(
        "--output-dir", default="data/processed",
        help="Parent directory for outputs (default: data/processed)"
    )
    args = parser.parse_args()
    run_comparison(args.input, args.bandwidths, args.output_dir)
