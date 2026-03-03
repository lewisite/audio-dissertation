"""
Flask Survey Application
========================
Dissertation: "The Effect of Explainability on End-User Trust in Neural Audio Codecs"

Participants:
  1. Give consent + demographics
  2. Work through 4 codec conditions in counterbalanced order
  3. For each condition: listen to audio, view results (Codec A shows explainability;
     Codec B shows nothing), then complete the 12-item Jian trust scale + MOS rating
  4. Receive completion code

CSV output is compatible with src/analyze_experiment.py.

Run:
    python app.py
    # Then open http://localhost:5000  (or use ngrok for remote access)
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # fix torch/numpy OpenMP clash on Windows
import sys
import uuid
import json
import csv
import time
import random
import shutil
import threading
from datetime import datetime
from pathlib import Path

from flask import (Flask, render_template, request, session,
                   redirect, url_for, jsonify, send_file, send_from_directory)
from werkzeug.utils import secure_filename

# ── Add src/ to Python path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'codec-trust-dissertation-2024')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024   # 32 MB max upload
app.config['TEMPLATES_AUTO_RELOAD'] = True             # Always serve fresh templates

# ── Directory layout ─────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER   = os.path.join(BASE_DIR, 'data', 'uploads')
PROCESSED_DIR   = os.path.join(BASE_DIR, 'static', 'audio', 'processed')
SAMPLES_DIR     = os.path.join(BASE_DIR, 'static', 'audio', 'samples')
RAW_DATA_DIR    = os.path.join(BASE_DIR, 'data', 'processed')
RESPONSES_FILE  = os.path.join(BASE_DIR, 'data', 'experiment_responses.csv')

for d in [UPLOAD_FOLDER, PROCESSED_DIR, SAMPLES_DIR, RAW_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# ── Experimental design ───────────────────────────────────────────────────────
# 2 × 2 within-subjects: Codec (A = explainable, B = black-box) × Latency (50 / 150 ms)
CONDITIONS = [
    {'id': 0, 'codec': 'A', 'latency_ms': 50,  'label': 'System Alpha', 'mode': 'Fast'},
    {'id': 1, 'codec': 'A', 'latency_ms': 150, 'label': 'System Alpha', 'mode': 'Standard'},
    {'id': 2, 'codec': 'B', 'latency_ms': 50,  'label': 'System Beta',  'mode': 'Fast'},
    {'id': 3, 'codec': 'B', 'latency_ms': 150, 'label': 'System Beta',  'mode': 'Standard'},
]

# 4×4 Latin square — 4 balanced condition orderings
CONDITION_ORDERS = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
]

# Track which order has been used least (for better balance across participants)
_order_counts_file = os.path.join(BASE_DIR, 'data', 'order_counts.json')

def _get_next_order_index():
    """Return the Latin-square order index with the fewest assignments."""
    if os.path.exists(_order_counts_file):
        with open(_order_counts_file) as f:
            counts = json.load(f)
    else:
        counts = [0, 0, 0, 0]
    idx = int(counts.index(min(counts)))
    counts[idx] += 1
    with open(_order_counts_file, 'w') as f:
        json.dump(counts, f)
    return idx

# ── Jian et al. (2000) Trust in Automation Scale ─────────────────────────────
TRUST_ITEMS = [
    {'id': 1,  'text': 'The system is deceptive.',
               'reverse': True},
    {'id': 2,  'text': 'The system behaves in an underhanded manner.',
               'reverse': True},
    {'id': 3,  'text': "I am suspicious of the system's intent, action, or outputs.",
               'reverse': True},
    {'id': 4,  'text': 'I am wary of the system.',
               'reverse': True},
    {'id': 5,  'text': "The system's actions will have a harmful or injurious outcome.",
               'reverse': True},
    {'id': 6,  'text': 'I am confident in the system.',
               'reverse': False},
    {'id': 7,  'text': 'The system provides security.',
               'reverse': False},
    {'id': 8,  'text': 'The system has integrity.',
               'reverse': False},
    {'id': 9,  'text': 'The system is dependable.',
               'reverse': False},
    {'id': 10, 'text': 'The system is reliable.',
               'reverse': False},
    {'id': 11, 'text': 'I can trust the system.',
               'reverse': False},
    {'id': 12, 'text': 'I am familiar with the system.',
               'reverse': False},
]

# ── CSV helpers ───────────────────────────────────────────────────────────────
CSV_HEADERS = (
    ['participant_id', 'condition_order', 'step', 'codec', 'latency_ms',
     'audio_clip_id', 'mos_rating', 'session_date',
     'age', 'gender', 'audio_background', 'hearing_impairment', 'headphones']
    + [f'trust_q{i}' for i in range(1, 13)]
)

def _ensure_csv():
    if not os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()

def _save_response(row: dict):
    _ensure_csv()
    with open(RESPONSES_FILE, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(
            {k: row.get(k, '') for k in CSV_HEADERS}
        )

# ── Background processing jobs ────────────────────────────────────────────────
_jobs: dict = {}        # job_id -> {status, result}
_jobs_lock = threading.Lock()

def _check_precomputed(clip_stem: str, codec: str, latency_ms: int) -> dict | None:
    """
    Return pre-computed result dict if prepare_samples.py has already processed
    this clip/condition, otherwise return None.
    """
    tag = f'{clip_stem}_{codec.lower()}_{latency_ms}ms'

    proc = os.path.join(PROCESSED_DIR, f'{tag}_reconstructed.wav')
    if not os.path.exists(proc):
        return None

    result = {
        'codec':         codec,
        'processed_url': f'/static/audio/processed/{tag}_reconstructed.wav',
    }

    # Original file size + duration (needed for file-size display for both A and B)
    orig_path = os.path.join(SAMPLES_DIR, f'{clip_stem}.wav')
    if os.path.exists(orig_path):
        result['original_size_bytes'] = os.path.getsize(orig_path)
        try:
            import soundfile as sf
            result['duration_s']     = round(sf.info(orig_path).duration, 2)
        except Exception:
            result['duration_s'] = 30.0
    result['bandwidth_kbps'] = 24.0   # both codecs always run at 24 kbps

    if codec == 'A':
        def _url(suffix):
            fn = os.path.join(PROCESSED_DIR, f'{tag}_{suffix}')
            return f'/static/audio/processed/{tag}_{suffix}' if os.path.exists(fn) else None

        resid = os.path.join(PROCESSED_DIR, f'{tag}_residual.wav')
        result['residual_url']    = f'/static/audio/processed/{tag}_residual.wav' if os.path.exists(resid) else None
        result['fig_waveforms']      = _url('waveforms.png')
        result['fig_spectrograms']   = _url('spectrograms.png')
        result['fig_heatmap']        = _url('heatmap.png')
        result['fig_stats']          = _url('stats.png')
        result['fig_input_analysis'] = _url('input_analysis.png')
        result['fig_saliency']       = _url('saliency.png')

        # Try to read summary from data/processed/
        summaries = sorted(
            Path(RAW_DATA_DIR).glob(f'*_bw24.0/summary.txt'),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        result['summary'] = summaries[0].read_text(encoding='utf-8', errors='replace') if summaries else ''

    # Read per-clip metrics JSON for BOTH codecs (saved alongside the reconstructed wav).
    # Never read from the global data/processed/ metrics.json — that always
    # returns the most-recently-run file, which is wrong for other clips.
    per_clip_metrics = os.path.join(PROCESSED_DIR, f'{tag}_metrics.json')
    if os.path.exists(per_clip_metrics):
        try:
            result['metrics'] = json.loads(
                open(per_clip_metrics, encoding='utf-8').read()
            )
        except Exception:
            pass

    return result


def _process_job(job_id: str, audio_path: str, condition: dict):
    """Run codec in a background thread. Updates _jobs[job_id] on completion."""
    with _jobs_lock:
        _jobs[job_id]['status'] = 'running'

    # Initialise so cleanup block can reference them even if an exception fires early
    was_trimmed  = False
    process_path = audio_path

    try:
        codec      = condition['codec']
        latency_ms = condition['latency_ms']
        clip_stem  = Path(audio_path).stem
        safe_stem  = f"{clip_stem}_{codec}_{latency_ms}ms"

        t0 = time.perf_counter()

        # ── Fast path: serve pre-computed results if available ───────────────
        precomp = _check_precomputed(clip_stem, codec, latency_ms)
        if precomp is not None:
            # Simulate target latency so the UI feels consistent
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms < latency_ms:
                time.sleep((latency_ms - elapsed_ms) / 1000)
            precomp['processing_time_ms'] = round((time.perf_counter() - t0) * 1000, 1)
            precomp['status'] = 'complete'
            with _jobs_lock:
                _jobs[job_id] = precomp
            return

        input_ext = Path(audio_path).suffix.lower().lstrip('.')
        try:
            import soundfile as sf
            orig_duration = float(sf.info(audio_path).duration)
        except Exception:
            orig_duration = None

        if codec == 'A':
            from pipeline import run_pipeline
            metrics, out_dir = run_pipeline(
                process_path, bandwidth=24.0,
                output_dir=os.path.join(BASE_DIR, 'data', 'processed')
            )
            out_dir = Path(out_dir)

            # ── Copy outputs to static/audio/processed/ for web serving ──────
            def _cp(src_name, dst_name):
                src = out_dir / src_name
                dst = os.path.join(PROCESSED_DIR, dst_name)
                if src.exists():
                    shutil.copy2(str(src), dst)
                    return f'/static/audio/processed/{dst_name}'
                return None

            proc_url     = _cp('reconstructed.wav',       f'{safe_stem}_reconstructed.wav')
            resid_url    = _cp('residual.wav',             f'{safe_stem}_residual.wav')
            fig_wave     = _cp('fig_waveforms.png',        f'{safe_stem}_waveforms.png')
            fig_spec     = _cp('fig_spectrograms.png',     f'{safe_stem}_spectrograms.png')
            fig_heatmap  = _cp('fig_codebook_heatmap.png',    f'{safe_stem}_heatmap.png')
            fig_stats    = _cp('fig_codebook_stats.png',      f'{safe_stem}_stats.png')
            fig_input    = _cp('fig_input_analysis.png',      f'{safe_stem}_input_analysis.png')
            fig_saliency = _cp('fig_saliency_comparison.png', f'{safe_stem}_saliency.png')
            summary_path = out_dir / 'summary.txt'
            summary_text = summary_path.read_text(encoding='utf-8', errors='replace') if summary_path.exists() else ''

            perc = metrics.get('perceptual', {}) or {}
            clip_metrics = {
                'snr_db':          round(metrics.get('waveform', {}).get('snr_db', 0), 2),
                'si_sdr_db':       round(metrics.get('waveform', {}).get('si_sdr_db', 0), 2),
                'mcd_db':          round(perc.get('mcd_db') or 0, 3),
                'mos_estimate':    round(perc.get('mos_estimate') or 0, 2),
                'bandwidth_kbps':  metrics.get('bandwidth_kbps', 24.0),
                'n_codebooks':     metrics.get('n_codebooks', 0),
                'n_frames':        metrics.get('n_frames', 0),
                'duration_s':      round(metrics.get('duration_seconds', 0), 2),
                'encode_time_s':   round(metrics.get('encode_time_seconds', 0), 3),
            }
            # Persist per-clip metrics so _check_precomputed() can serve them
            # correctly on future requests without touching the stale global metrics.json
            try:
                per_clip_path = os.path.join(PROCESSED_DIR, f'{safe_stem}_metrics.json')
                with open(per_clip_path, 'w', encoding='utf-8') as mf:
                    json.dump(clip_metrics, mf)
            except Exception:
                pass
            result = {
                'codec':               'A',
                'was_trimmed':         was_trimmed,
                'original_duration_s': orig_duration,
                'input_format':        input_ext,
                'processed_url':       proc_url,
                'residual_url':        resid_url,
                'fig_waveforms':       fig_wave,
                'fig_spectrograms':    fig_spec,
                'fig_heatmap':         fig_heatmap,
                'fig_stats':           fig_stats,
                'fig_input_analysis':  fig_input,
                'fig_saliency':        fig_saliency,
                'input_analysis':      metrics.get('input_analysis', {}),
                'summary':             summary_text,
                'original_size_bytes': os.path.getsize(audio_path),
                'metrics':             clip_metrics,
            }

        else:  # codec == 'B'
            from codec_b import run_codec_b
            _, out_dir = run_codec_b(
                process_path, bandwidth=24.0,
                output_dir=os.path.join(BASE_DIR, 'data', 'processed')
            )
            out_dir = Path(out_dir)
            proc_name = f'{safe_stem}_reconstructed.wav'
            dst       = os.path.join(PROCESSED_DIR, proc_name)
            shutil.copy2(str(out_dir / 'reconstructed.wav'), dst)

            try:
                import soundfile as sf
                _dur = round(sf.info(process_path).duration, 2)
            except Exception:
                _dur = 30.0

            result = {
                'codec':               'B',
                'was_trimmed':         was_trimmed,
                'original_duration_s': orig_duration,
                'input_format':        input_ext,
                'processed_url':       f'/static/audio/processed/{proc_name}',
                'original_size_bytes': os.path.getsize(audio_path),
                'duration_s':          _dur,
                'bandwidth_kbps':      24.0,
            }

        # ── Simulate target latency (pad if codec was faster) ─────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms < latency_ms:
            time.sleep((latency_ms - elapsed_ms) / 1000)

        result['processing_time_ms'] = round((time.perf_counter() - t0) * 1000, 1)
        result['status'] = 'complete'

    except Exception as exc:
        result = {'status': 'error', 'error': str(exc)}

    with _jobs_lock:
        _jobs[job_id] = result

    # Delete uploaded file — not retained after processing
    if os.path.normpath(audio_path).startswith(os.path.normpath(UPLOAD_FOLDER)):
        try:
            os.remove(audio_path)
        except OSError:
            pass

# ── Sample audio files ────────────────────────────────────────────────────────
def _get_samples():
    """Return list of {id, name, url} for available sample audio files."""
    samples = []
    for f in sorted(Path(SAMPLES_DIR).glob('*.wav')):
        stem = f.stem
        samples.append({
            'id':   stem,
            'name': stem.replace('_', ' ').title(),
            'url':  f'/static/audio/samples/{f.name}',
        })
    # If no samples prepared, fall back to the bundled sample
    if not samples:
        fallback = os.path.join(SAMPLES_DIR, 'now_stand_aside.wav')
        if os.path.exists(fallback):
            samples.append({'id': 'now_stand_aside', 'name': 'Now Stand Aside',
                             'url': '/static/audio/samples/now_stand_aside.wav'})
    return samples

def _allowed(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/start', methods=['POST'])
def start():
    order_idx      = _get_next_order_index()
    condition_order = CONDITION_ORDERS[order_idx]
    participant_id  = str(uuid.uuid4())[:8].upper()

    session.clear()
    session['participant_id']  = participant_id
    session['condition_order'] = condition_order
    session['start_time']      = datetime.now().isoformat()

    return redirect(url_for('demographics'))


@app.route('/demographics', methods=['GET', 'POST'])
def demographics():
    if 'participant_id' not in session:
        return redirect(url_for('welcome'))

    if request.method == 'POST':
        session['demographics'] = {
            'age':               request.form.get('age', ''),
            'gender':            request.form.get('gender', ''),
            'audio_background':  request.form.get('audio_background', ''),
            'hearing_impairment':request.form.get('hearing_impairment', 'no'),
            'headphones':        request.form.get('headphones', ''),
        }
        return redirect(url_for('audio_test', step=0))

    return render_template('demographics.html',
                           participant_id=session['participant_id'])


@app.route('/test/<int:step>')
def audio_test(step):
    if 'participant_id' not in session:
        return redirect(url_for('welcome'))
    if step >= len(CONDITIONS):
        return redirect(url_for('complete'))

    order     = session['condition_order']
    condition = CONDITIONS[order[step]]
    samples   = _get_samples()

    return render_template('audio_test.html',
                           step=step,
                           total_steps=len(CONDITIONS),
                           condition=condition,
                           samples=samples,
                           trust_items=TRUST_ITEMS,
                           participant_id=session['participant_id'])


@app.route('/api/process', methods=['POST'])
def api_process():
    """Start background codec processing job. Returns {job_id}."""
    if 'participant_id' not in session:
        return jsonify({'error': 'Session expired — please reload the page.'}), 401

    step      = int(request.form.get('step', 0))
    source    = request.form.get('source', 'sample')   # 'sample' | 'upload'
    sample_id = request.form.get('sample_id', '')

    order     = session['condition_order']
    condition = CONDITIONS[order[step]]

    # ── Resolve input audio path ──────────────────────────────────────────────
    if source == 'upload':
        f = request.files.get('audio_file')
        if not f or not _allowed(f.filename):
            return jsonify({'error': 'Please upload a valid audio file (WAV/MP3/FLAC).'}), 400
        filename = secure_filename(f.filename)
        audio_path = os.path.join(UPLOAD_FOLDER,
                                  f"{session['participant_id']}_{filename}")
        f.save(audio_path)
        # Truncate uploaded files to 2 minutes to keep processing time reasonable
        upload_warning = None
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            if info.duration > 120:
                data, sr = sf.read(audio_path)
                max_samples = int(120 * sr)
                sf.write(audio_path, data[:max_samples], sr)
                upload_warning = f'Your clip was {info.duration:.0f}s — we trimmed it to the first 2 minutes.'
        except Exception:
            pass
    else:
        audio_path = os.path.join(SAMPLES_DIR, f'{sample_id}.wav')
        if not os.path.exists(audio_path):
            # Fall back to bundled sample
            audio_path = os.path.join(SAMPLES_DIR, 'now_stand_aside.wav')
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Sample file not found. Run prepare_samples.py first.'}), 404

    # ── Launch background job ─────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {'status': 'pending'}

    t = threading.Thread(target=_process_job,
                         args=(job_id, audio_path, condition), daemon=True)
    t.start()

    session['last_clip_id'] = Path(audio_path).stem
    resp = {'job_id': job_id}
    if source == 'upload' and upload_warning:
        resp['warning'] = upload_warning
    return jsonify(resp)


@app.route('/api/status/<job_id>')
def api_status(job_id):
    """Poll processing job status."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)


@app.route('/submit/<int:step>', methods=['POST'])
def submit_step(step):
    """Save one condition's responses to CSV, redirect to next step."""
    if 'participant_id' not in session:
        return redirect(url_for('welcome'))

    order     = session['condition_order']
    condition = CONDITIONS[order[step]]

    trust = {f'trust_q{i}': request.form.get(f'trust_q{i}', '')
             for i in range(1, 13)}

    demo = session.get('demographics', {})

    _save_response({
        'participant_id':   session['participant_id'],
        'condition_order':  ','.join(str(x) for x in order),
        'step':             step,
        'codec':            condition['codec'],
        'latency_ms':       condition['latency_ms'],
        'audio_clip_id':    request.form.get('audio_clip_id',
                                             session.get('last_clip_id', '')),
        'mos_rating':       request.form.get('mos_rating', ''),
        'session_date':     datetime.now().strftime('%Y-%m-%d'),
        'age':              demo.get('age', ''),
        'gender':           demo.get('gender', ''),
        'audio_background': demo.get('audio_background', ''),
        'hearing_impairment': demo.get('hearing_impairment', ''),
        'headphones':       demo.get('headphones', ''),
        **trust,
    })

    next_step = step + 1
    if next_step >= len(CONDITIONS):
        return redirect(url_for('complete'))
    return redirect(url_for('audio_test', step=next_step))


@app.route('/complete')
def complete():
    if 'participant_id' not in session:
        return redirect(url_for('welcome'))
    pid  = session['participant_id']
    code = f'CODEC-{pid}'
    return render_template('complete.html',
                           participant_id=pid, completion_code=code)


@app.route('/admin/responses')
def admin_download():
    """Download the CSV of all responses (protect with ADMIN_KEY env var)."""
    key = request.args.get('key', '')
    if key != os.environ.get('ADMIN_KEY', 'admin2024'):
        return 'Unauthorized', 401
    if not os.path.exists(RESPONSES_FILE):
        return 'No data collected yet.', 404
    return send_file(RESPONSES_FILE, as_attachment=True,
                     download_name='experiment_responses.csv',
                     mimetype='text/csv')


# ── Entry point ───────────────────────────────────────────────────────────────
def _kill_port(port: int):
    """Kill any processes already listening on *port* before we bind to it."""
    import subprocess, os as _os
    try:
        r = subprocess.run(
            ['netstat', '-ano'], capture_output=True, text=True, shell=True
        )
        pids = set()
        for line in r.stdout.splitlines():
            if f':{port}' in line and 'LISTEN' in line:
                parts = line.split()
                if parts:
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
        own_pid = _os.getpid()
        for pid in pids:
            if pid == own_pid or pid == 0:
                continue
            subprocess.run(
                ['taskkill', '/F', '/PID', str(pid)],
                capture_output=True, shell=True
            )
            print(f"  Killed stale process PID {pid} on port {port}")
    except Exception as e:
        print(f"  Warning: could not clear port {port}: {e}")


if __name__ == '__main__':
    _ensure_csv()
    print("\n" + "=" * 60)
    print("  Codec Trust Survey — starting server")
    print("  Local:   http://localhost:5000")
    print("  Network: http://0.0.0.0:5000")
    print("  Admin:   http://localhost:5000/admin/responses?key=admin2024")
    print("=" * 60 + "\n")
    port = int(os.environ.get('PORT', 5000))
    _kill_port(port)
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
