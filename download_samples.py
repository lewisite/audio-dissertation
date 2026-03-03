"""
download_samples.py
===================
Downloads famous public-domain audio clips and prepares them as 30-second
24 kHz mono WAV files ready for the codec survey.

All sources are public domain:
  - NASA / US Government recordings  (Greatest Speeches, archive.org)
  - Toscanini / NBC Symphony, 1939   (copyright expired, pre-1928 rule)
  - Apollo 8, Christmas 1968         (NASA public domain)

Run:
    python download_samples.py
    python prepare_samples.py      <- process through all 4 codec conditions
    python app.py                  <- start the survey
"""

import os
import sys
import urllib.request
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'audio', 'samples')
os.makedirs(SAMPLES_DIR, exist_ok=True)

TARGET_SR       = 24_000   # EnCodec native sample rate
CLIP_DURATION_S = 30       # seconds to keep per sample

# ── Resolve ffmpeg (uses the bundled imageio-ffmpeg binary) ───────────────────
def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    # Fall back to system ffmpeg
    for candidate in ['ffmpeg', 'ffmpeg.exe']:
        r = subprocess.run([candidate, '-version'], capture_output=True)
        if r.returncode == 0:
            return candidate
    raise RuntimeError(
        "ffmpeg not found. Install it with:\n"
        "    pip install imageio-ffmpeg\n"
        "or  conda install -c conda-forge ffmpeg"
    )

FFMPEG = _get_ffmpeg()
print(f"Using ffmpeg: {FFMPEG}")

# ── Clip definitions ──────────────────────────────────────────────────────────
# All from archive.org — confirmed public-domain sources.
CLIPS = [
    {
        'id':      'moon_landing',
        'name':    "Moon Landing — Neil Armstrong, July 1969",
        'url':     'https://ia903402.us.archive.org/6/items/'
                   'Greatest_Speeches_of_the_20th_Century/TheMoonLanding.mp3',
        'start_s': 0,    # "The Eagle has landed" / "one small step" near the start
        'note':    'Iconic NASA recording — public domain (US Government work)',
    },
    {
        'id':      'jfk_inaugural',
        'name':    'JFK Inaugural Address — "Ask not what your country…" (1961)',
        'url':     'https://ia903402.us.archive.org/6/items/'
                   'Greatest_Speeches_of_the_20th_Century/InauguralAddress-1961.mp3',
        'start_s': 0,    # Opens with "Let the word go forth from this time and place"
        'note':    'US Government presidential address — public domain',
    },
    {
        'id':      'beethoven_5th',
        'name':    "Beethoven's 5th Symphony — Toscanini / NBC, 1939",
        'url':     'https://ia801207.us.archive.org/16/items/'
                   'BeethovenSymphonyNo.5/ToscaniniBeethoven5_64kb.mp3',
        'start_s': 0,    # Famous da-da-da-DUM opening is the very first seconds
        'note':    'Pre-1928 recording — copyright expired, public domain (US)',
    },
    {
        'id':      'christmas_from_space',
        'name':    "Apollo 8: Christmas Greeting from Space (1968)",
        'url':     'https://ia903402.us.archive.org/6/items/'
                   'Greatest_Speeches_of_the_20th_Century/ChristmasGreetingfromSpace.mp3',
        'start_s': 0,
        'note':    'NASA Apollo 8 mission audio — public domain (US Government work)',
    },
]


# ── Download helpers ──────────────────────────────────────────────────────────
class _ProgressBar:
    def __init__(self, label):
        self._label = label
        self._last_pct = -1
    def __call__(self, count, block_size, total):
        if total <= 0:
            return
        pct = min(100, count * block_size * 100 // total)
        if pct >= self._last_pct + 5 or pct == 100:
            mb = count * block_size / 1_048_576
            print(f"\r    [{self._label}] {pct:3d}%  ({mb:.1f} MB)   ",
                  end='', flush=True)
            self._last_pct = pct


def _ffmpeg_to_wav(src: str, dst: str, start_s: float, duration_s: float):
    """
    Use ffmpeg to:
      - seek to start_s
      - extract duration_s seconds
      - mix down to mono (-ac 1)
      - resample to TARGET_SR
      - save as 16-bit PCM WAV
    """
    cmd = [
        FFMPEG, '-y',
        '-ss', str(start_s),
        '-i', src,
        '-t', str(duration_s),
        '-ac', '1',                     # mono
        '-ar', str(TARGET_SR),          # 24 kHz
        '-sample_fmt', 's16',           # 16-bit PCM
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed:\n{result.stderr[-800:]}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 62)
    print("  Downloading famous public-domain audio samples")
    print("=" * 62)
    print(f"  Output directory: {SAMPLES_DIR}\n")

    ok_count = 0
    for clip in CLIPS:
        cid  = clip['id']
        name = clip['name']
        url  = clip['url']
        out  = os.path.join(SAMPLES_DIR, f'{cid}.wav')

        print(f"  {name}")
        print(f"  {clip['note']}")

        if os.path.exists(out):
            info = sf.info(out)
            print(f"  Already downloaded — {info.duration:.1f}s @ {info.samplerate} Hz")
            print()
            ok_count += 1
            continue

        # ── Download to temp file ─────────────────────────────────────────────
        ext  = Path(url).suffix or '.mp3'
        tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.close()

        print(f"  Downloading: {url.split('/')[-1]}")
        try:
            urllib.request.urlretrieve(
                url, tmp.name, reporthook=_ProgressBar(cid))
            print()
        except Exception as exc:
            print(f"\n  ERROR downloading: {exc}")
            try: os.unlink(tmp.name)
            except OSError: pass
            print()
            continue

        # ── Convert MP3 → trimmed 24 kHz mono WAV via ffmpeg ─────────────────
        try:
            print(f"  Converting to 30s mono 24 kHz WAV ...", end=' ', flush=True)
            _ffmpeg_to_wav(tmp.name, out, clip['start_s'], CLIP_DURATION_S)
            info     = sf.info(out)
            size_kb  = os.path.getsize(out) // 1024
            print(f"OK  ({info.duration:.1f}s, {size_kb} KB)")
            ok_count += 1
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 62)
    print(f"  {ok_count}/{len(CLIPS)} samples ready in: {SAMPLES_DIR}")
    if ok_count > 0:
        print()
        print("  Next steps:")
        print("    python prepare_samples.py   <- process through all 4 codecs")
        print("    python app.py               <- start the survey server")
    print("=" * 62 + "\n")


if __name__ == '__main__':
    main()
