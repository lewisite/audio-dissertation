"""
Codec B: EnCodec as a Black Box (No Explainability)
=====================================================
Dissertation control condition for RQ1.

Same underlying neural codec as Codec A (EnCodec 24 kHz), but with zero
explainability features surfaced to the user. The user receives only the
reconstructed audio — no summary, no residual playback, no visualisations,
no codebook analysis. This simulates the experience of a standard neural
codec as a black box.

The internal metrics ARE computed and saved (for researcher use in the
comparison analysis), but they are never shown to the end-user during
the experimental session — that is the key design distinction from Codec A.

Usage:
  python src/codec_b.py --input test.wav --bandwidth 6.0
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (
    load_audio, save_audio, encode_audio, decode_audio,
    compute_waveform_metrics, compute_spectral_metrics, compute_perceptual_metrics,
    analyze_tokens, postprocess_audio,
)
from encodec import EncodecModel


def run_codec_b(input_path: str, bandwidth: float = 24.0,
                output_dir: str = "data/processed") -> tuple:
    """
    Codec B: silent encode-decode with no user-facing explanation.

    User-facing output  : reconstructed.wav only
    Researcher output   : hidden_metrics.json (never shown to participant)

    Returns (hidden_metrics dict, output_directory Path)
    """
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_codecB_bw{bandwidth}"
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    audio_np, src_sr = load_audio(input_path)
    duration = audio_np.shape[-1] / src_sr

    # ── Encode ───────────────────────────────────────────────────────────────
    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(bandwidth)
    model.eval()
    model.segment = None   # full-clip pass — no chunk-boundary artefacts

    t0 = time.perf_counter()
    frames, wav_tensor, codes_np = encode_audio(model, audio_np, src_sr)
    encode_time = time.perf_counter() - t0

    # ── Decode ───────────────────────────────────────────────────────────────
    recon_np = decode_audio(model, frames)
    # Post-process: restore high-frequency brightness after codec rolloff
    recon_np = postprocess_audio(recon_np, model.sample_rate)

    orig_at_model_sr = wav_tensor[0].cpu().numpy()
    T = min(orig_at_model_sr.shape[-1], recon_np.shape[-1])
    orig_aligned  = orig_at_model_sr[..., :T]
    recon_aligned = recon_np[..., :T]

    # ── Save user-facing output (reconstructed audio ONLY) ───────────────────
    save_audio(str(out_dir / "reconstructed.wav"), recon_aligned, model.sample_rate)

    # ── Compute metrics (for researcher comparison — NOT shown to user) ───────
    wm = compute_waveform_metrics(orig_aligned, recon_aligned)
    sm = compute_spectral_metrics(orig_aligned, recon_aligned, model.sample_rate)
    pm = compute_perceptual_metrics(orig_aligned, recon_aligned, model.sample_rate)
    tm = analyze_tokens(codes_np)
    n_q, n_frames = codes_np.shape

    hidden_metrics = {
        "run_id":              run_id,
        "codec":               "B",
        "description":         "EnCodec black box — no explainability features",
        "input_file":          str(input_path),
        "bandwidth_kbps":      bandwidth,
        "src_sample_rate":     src_sr,
        "model_sample_rate":   model.sample_rate,
        "duration_seconds":    float(duration),
        "encode_time_seconds": float(encode_time),
        "n_codebooks":         n_q,
        "n_frames":            n_frames,
        "waveform":            wm,
        "spectral":            sm,
        "perceptual":          pm,
        "tokens":              tm,
        "user_facing_outputs": ["reconstructed.wav"],
        "explainability_features_shown": False,
    }

    # Saved as "hidden_metrics" to make the design intent explicit in filenames
    with open(out_dir / "hidden_metrics.json", "w") as f:
        json.dump(hidden_metrics, f, indent=2)

    return hidden_metrics, out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Codec B: EnCodec black box (no explainability) — dissertation control condition"
    )
    parser.add_argument("--input",     required=True,  help="Path to input audio file (.wav)")
    parser.add_argument("--bandwidth", type=float, default=24.0,
                        choices=[1.5, 3.0, 6.0, 12.0, 24.0],
                        help="Target bandwidth in kbps (default: 6.0)")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Parent directory for outputs (default: data/processed)")
    args = parser.parse_args()

    metrics, out_dir = run_codec_b(args.input, args.bandwidth, args.output_dir)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  CODEC B  |  BLACK BOX OUTPUT")
    print(sep)
    print(f"  Input     : {args.input}")
    print(f"  Bandwidth : {args.bandwidth} kbps")
    print(f"  Output    : {out_dir}")
    print(f"\n  User sees : reconstructed.wav only (no explanation)")
    print(f"\n  [Researcher-only metrics — not shown to participant]")
    print(f"  Encode time : {metrics['encode_time_seconds']:.3f}s")
    print(f"  SNR         : {metrics['waveform']['snr_db']:.1f} dB")
    print(f"  SI-SDR      : {metrics['waveform']['si_sdr_db']:.1f} dB")
    print(f"  Spectral SNR: {metrics['spectral']['spectral_snr_db']:.1f} dB")
    print(sep + "\n")
