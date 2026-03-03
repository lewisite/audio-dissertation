"""
Codec Opus: Traditional Codec Baseline
=======================================
Dissertation traditional baseline using the Opus codec via pyogg + libopusenc.

Opus is an open, royalty-free traditional codec standardised as RFC 6716.
It is the closest analogue to EnCodec in terms of target bitrates and use
cases, and is used in WebRTC, Discord, and most VoIP applications — making
it the natural "state of the art traditional codec" baseline for RQ2.

How it differs from EnCodec (Codec A/B):
  - Compression algorithm : psychoacoustic masking + MDCT (Opus) vs
                            neural autoencoder + RVQ (EnCodec)
  - Transparency          : rule-based bit allocation is more auditable in
                            principle, but provides no user-facing explanation
  - Quality at low bitrate: EnCodec outperforms Opus below ~12 kbps

Bitrate guidance (Opus mono):
  6  kbps — minimum usable; significant artefacts
  12 kbps — comparable to EnCodec 6 kbps perceptually
  24 kbps — good speech quality
  48 kbps — near-transparent for most speech content
  96 kbps — high-fidelity music

Note: Opus always encodes/decodes at 48 kHz internally.
      Audio is resampled to 24 kHz after decoding for metric comparison.

Usage:
  python src/codec_opus.py --input test.wav --bitrate 12000
"""

import argparse
import ctypes
import json
import time
from datetime import datetime
from pathlib import Path
import tempfile
import os

import numpy as np
from scipy.signal import resample_poly

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (
    load_audio, save_audio,
    compute_waveform_metrics, compute_spectral_metrics,
)

try:
    import pyogg
    import pyogg.opus as op
    OPUS_AVAILABLE = True
except ImportError:
    OPUS_AVAILABLE = False


# Opus CTL constant for setting bitrate
_OPUS_SET_BITRATE_REQUEST = 4002

# Valid Opus frame sizes at 24 kHz (samples)
# Corresponds to 2.5/5/10/20/40/60 ms at 48 kHz, halved for 24 kHz input
_FRAME_SIZE = 960   # 40 ms at 24 kHz — good balance of latency and quality


def _encode_decode_opus(audio_mono: np.ndarray, src_sr: int, bitrate_bps: int) -> np.ndarray:
    """
    Encode audio through Opus at the given bitrate, decode back to PCM.

    Args:
        audio_mono  : 1-D float32 array, range [-1, 1]
        src_sr      : source sample rate (24000 recommended for EnCodec comparison)
        bitrate_bps : target bitrate in bits-per-second (e.g. 12000 = 12 kbps)

    Returns:
        decoded float32 array at src_sr, same approximate length as input
    """
    if not OPUS_AVAILABLE:
        raise RuntimeError("pyogg is not installed. Run: pip install pyogg")

    # Pad to a multiple of frame size so no samples are dropped
    remainder = len(audio_mono) % _FRAME_SIZE
    if remainder != 0:
        pad_len = _FRAME_SIZE - remainder
        audio_padded = np.concatenate([audio_mono, np.zeros(pad_len, dtype=np.float32)])
    else:
        audio_padded = audio_mono.copy()

    # ── Write OGG/Opus file ──────────────────────────────────────────────────
    tmpfile = tempfile.mktemp(suffix=".opus")
    comments  = op.ope_comments_create()
    error     = ctypes.c_int(0)

    enc = op.ope_encoder_create_file(
        tmpfile.encode("utf-8"),
        comments,
        ctypes.c_int(src_sr),
        ctypes.c_int(1),      # channels = 1 (mono)
        ctypes.c_int(0),      # family  = 0 (single stream)
        ctypes.byref(error),
    )
    if error.value != 0 or enc is None:
        op.ope_comments_destroy(comments)
        raise RuntimeError(f"ope_encoder_create_file failed (error {error.value})")

    op.ope_encoder_ctl(enc, _OPUS_SET_BITRATE_REQUEST, ctypes.c_int(bitrate_bps))

    # Write frame by frame
    for i in range(0, len(audio_padded), _FRAME_SIZE):
        frame = audio_padded[i : i + _FRAME_SIZE].astype(np.float32)
        ptr   = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        op.ope_encoder_write_float(enc, ptr, ctypes.c_int(_FRAME_SIZE))

    op.ope_encoder_drain(enc)
    op.ope_encoder_destroy(enc)
    op.ope_comments_destroy(comments)

    # ── Read back (Opus always decodes to 48 kHz) ───────────────────────────
    opus_file = pyogg.OpusFile(tmpfile)
    os.unlink(tmpfile)

    n_samples  = opus_file.buffer_length // 2   # int16 = 2 bytes per sample
    pcm_int16  = np.frombuffer(
        ctypes.string_at(opus_file.buffer, opus_file.buffer_length),
        dtype=np.int16,
    )
    pcm_float  = pcm_int16.astype(np.float32) / 32768.0

    # Resample from Opus internal 48 kHz back to src_sr for fair comparison
    opus_sr = opus_file.frequency  # always 48000
    if opus_sr != src_sr:
        # resample_poly uses integer up/down ratios
        from math import gcd
        g    = gcd(src_sr, opus_sr)
        up   = src_sr  // g
        down = opus_sr // g
        pcm_float = resample_poly(pcm_float, up, down).astype(np.float32)

    return pcm_float


def run_codec_opus(input_path: str, bitrate_bps: int = 12000,
                   output_dir: str = "data/processed") -> tuple:
    """
    Run the Opus codec on an audio file and compute all comparison metrics.

    Args:
        input_path  : path to input .wav file
        bitrate_bps : Opus target bitrate in bps (e.g. 12000 = 12 kbps)
        output_dir  : parent directory for outputs

    Returns:
        (metrics dict, output_directory Path)
    """
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_opus_{bitrate_bps//1000}kbps"
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  CODEC OPUS  |  TRADITIONAL BASELINE")
    print(sep)
    print(f"  Input   : {input_path}")
    print(f"  Bitrate : {bitrate_bps/1000:.1f} kbps")
    print(f"  Output  : {out_dir}")
    print()

    # ── Load audio ───────────────────────────────────────────────────────────
    print("[1/4] Loading audio...")
    audio_np, src_sr = load_audio(input_path)
    duration = audio_np.shape[-1] / src_sr
    print(f"      Duration : {duration:.2f}s  |  SR : {src_sr} Hz")

    # Use mono for Opus (average channels if stereo)
    if audio_np.shape[0] > 1:
        audio_mono = audio_np.mean(axis=0)
    else:
        audio_mono = audio_np[0]

    # ── Encode + Decode ──────────────────────────────────────────────────────
    print("[2/4] Encoding and decoding via Opus...")
    t0 = time.perf_counter()
    recon_mono = _encode_decode_opus(audio_mono, src_sr, bitrate_bps)
    encode_time = time.perf_counter() - t0
    print(f"      Encode+decode time : {encode_time:.3f}s")

    # Align lengths for comparison
    T = min(len(audio_mono), len(recon_mono))
    orig_aligned  = audio_mono[:T].reshape(1, -1)    # [1, T]
    recon_aligned = recon_mono[:T].reshape(1, -1)    # [1, T]
    residual_np   = orig_aligned - recon_aligned

    # ── Save outputs ─────────────────────────────────────────────────────────
    save_audio(str(out_dir / "reconstructed.wav"), recon_aligned, src_sr)
    save_audio(str(out_dir / "residual.wav"),      residual_np,   src_sr)
    print(f"      Saved reconstructed.wav and residual.wav")

    # ── Compute metrics ──────────────────────────────────────────────────────
    print("[3/4] Computing metrics...")
    wm = compute_waveform_metrics(orig_aligned, recon_aligned)
    sm = compute_spectral_metrics(orig_aligned, recon_aligned, src_sr)

    metrics = {
        "run_id":              run_id,
        "codec":               "Opus",
        "description":         "Traditional codec — Opus RFC 6716 via libopusenc",
        "input_file":          str(input_path),
        "bitrate_bps":         bitrate_bps,
        "bitrate_kbps":        bitrate_bps / 1000,
        "src_sample_rate":     src_sr,
        "opus_internal_sr":    48000,
        "duration_seconds":    float(duration),
        "encode_time_seconds": float(encode_time),
        "waveform":            wm,
        "spectral":            sm,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────────
    print("[4/4] Done.")
    print()
    print(f"  SNR          : {wm['snr_db']:.1f} dB")
    print(f"  SI-SDR       : {wm['si_sdr_db']:.1f} dB")
    print(f"  Spectral SNR : {sm['spectral_snr_db']:.1f} dB")
    print(f"  LSD          : {sm['log_spectral_distance']:.3f}")
    print(f"  Residual peak: {wm['residual_peak']:.4f}")
    print(f"\n  Outputs saved to: {out_dir}")
    print(sep + "\n")

    return metrics, out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Opus codec baseline — dissertation traditional codec comparison"
    )
    parser.add_argument("--input",   required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--bitrate", type=int, default=12000,
                        help="Opus target bitrate in bps (default: 12000 = 12 kbps)")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Parent directory for outputs (default: data/processed)")
    args = parser.parse_args()
    run_codec_opus(args.input, args.bitrate, args.output_dir)
