"""
prepare_samples.py
==================
Pre-processes sample audio files through all 4 experimental conditions so
the survey loads instantly without waiting for on-the-fly codec inference.

Run once before launching the survey:

    python prepare_samples.py

Outputs go to:
    static/audio/samples/   — copies of original WAV files
    static/audio/processed/ — codec outputs (reconstructed + residual + figures)
    data/processed/         — full pipeline outputs (metrics, summary, etc.)

Conditions processed:
  codec_a_50ms   — Codec A (explainable EnCodec), 50 ms simulated latency
  codec_a_150ms  — Codec A (explainable EnCodec), 150 ms simulated latency
  codec_b_50ms   — Codec B (black-box EnCodec), 50 ms simulated latency
  codec_b_150ms  — Codec B (black-box EnCodec), 150 ms simulated latency
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SRC_DIR       = os.path.join(BASE_DIR, 'src')
SAMPLES_DIR   = os.path.join(BASE_DIR, 'static', 'audio', 'samples')
PROCESSED_DIR = os.path.join(BASE_DIR, 'static', 'audio', 'processed')
RAW_DATA_DIR  = os.path.join(BASE_DIR, 'data', 'processed')

sys.path.insert(0, SRC_DIR)

for d in [SAMPLES_DIR, PROCESSED_DIR, RAW_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

CONDITIONS = [
    {'codec': 'A', 'latency_ms': 50},
    {'codec': 'A', 'latency_ms': 150},
    {'codec': 'B', 'latency_ms': 50},
    {'codec': 'B', 'latency_ms': 150},
]


def process_sample(audio_path: str, verbose: bool = True):
    """Process one audio file through all 4 conditions."""
    stem = Path(audio_path).stem
    if verbose:
        print(f"\n{'='*58}")
        print(f"  Sample: {stem}")
        print('='*58)

    # ── Copy original to samples directory (skip if already there) ───────────
    sample_dst = os.path.join(SAMPLES_DIR, f'{stem}.wav')
    if os.path.abspath(audio_path) != os.path.abspath(sample_dst):
        shutil.copy2(audio_path, sample_dst)
        if verbose:
            print(f"  Copied original -> {sample_dst}")
    else:
        if verbose:
            print(f"  Original already in samples dir: {sample_dst}")

    # ── Process through each condition ────────────────────────────────────────
    for cond in CONDITIONS:
        codec      = cond['codec']
        latency_ms = cond['latency_ms']
        tag        = f'{stem}_{codec.lower()}_{latency_ms}ms'

        if verbose:
            print(f"\n  Condition: Codec {codec} | {latency_ms} ms")

        try:
            if codec == 'A':
                from pipeline import run_pipeline
                metrics, out_dir = run_pipeline(
                    audio_path, bandwidth=24.0, output_dir=RAW_DATA_DIR
                )
                out_dir = Path(out_dir)

                def _cp(src_name, dst_name):
                    src = out_dir / src_name
                    if src.exists():
                        dst = os.path.join(PROCESSED_DIR, dst_name)
                        shutil.copy2(str(src), dst)
                        if verbose:
                            print(f"    Saved {dst_name}")

                _cp('reconstructed.wav',       f'{tag}_reconstructed.wav')
                _cp('residual.wav',             f'{tag}_residual.wav')
                _cp('fig_waveforms.png',        f'{tag}_waveforms.png')
                _cp('fig_spectrograms.png',     f'{tag}_spectrograms.png')
                _cp('fig_codebook_heatmap.png', f'{tag}_heatmap.png')
                _cp('fig_codebook_stats.png',   f'{tag}_stats.png')

            else:  # codec == 'B'
                from codec_b import run_codec_b
                _, out_dir = run_codec_b(
                    audio_path, bandwidth=24.0, output_dir=RAW_DATA_DIR
                )
                out_dir = Path(out_dir)
                src = out_dir / 'reconstructed.wav'
                if src.exists():
                    dst_name = f'{tag}_reconstructed.wav'
                    shutil.copy2(str(src), os.path.join(PROCESSED_DIR, dst_name))
                    if verbose:
                        print(f"    Saved {dst_name}")

        except Exception as exc:
            print(f"    ERROR processing Codec {codec} / {latency_ms} ms: {exc}")
            import traceback
            traceback.print_exc()

    if verbose:
        print(f"\n  Done: {stem}")


def main():
    parser = argparse.ArgumentParser(
        description='Pre-process sample audio for the survey.'
    )
    parser.add_argument(
        '--input', nargs='+',
        help='Audio file(s) to process. Defaults to test.wav if not specified.'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output.'
    )
    args = parser.parse_args()

    # Default: use test.wav
    if args.input:
        audio_files = args.input
    else:
        default = os.path.join(BASE_DIR, 'test.wav')
        if not os.path.exists(default):
            print(f"ERROR: default sample {default} not found.")
            print("Usage: python prepare_samples.py --input path/to/audio.wav")
            sys.exit(1)
        audio_files = [default]

    print(f"\nPre-processing {len(audio_files)} audio file(s) x {len(CONDITIONS)} conditions")
    print(f"Output directory: {PROCESSED_DIR}\n")

    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"WARNING: {audio_file} not found, skipping.")
            continue
        process_sample(audio_file, verbose=not args.quiet)

    print(f"\n{'='*58}")
    print("  Pre-processing complete!")
    print(f"  Samples:   {SAMPLES_DIR}")
    print(f"  Processed: {PROCESSED_DIR}")
    print(f"  Now launch the survey:  python app.py")
    print('='*58 + '\n')


if __name__ == '__main__':
    main()
