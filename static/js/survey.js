/**
 * survey.js — Frontend logic for the Codec Trust Survey
 */

'use strict';

/* ── Utility helpers ────────────────────────────────────────────────────────── */
function formatBytes(bytes) {
  if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + ' MB';
  if (bytes >= 1024)    return (bytes / 1024).toFixed(0) + ' KB';
  return bytes + ' B';
}

function generateNarrative(m, processingTimeMs, origBytes) {
  const snr    = parseFloat(m.snr_db)         || 0;
  const sidr   = parseFloat(m.si_sdr_db)      || 0;
  const bw     = parseFloat(m.bandwidth_kbps) || 6;
  const dur    = parseFloat(m.duration_s)     || 30;
  const cb     = parseInt(m.n_codebooks)      || 8;
  const frames = parseInt(m.n_frames)         || 0;

  // ── 1. What happened ─────────────────────────────────────────────────────
  const compressedBytes = Math.round(bw * 1000 / 8 * dur);
  const ratio           = origBytes ? Math.round(origBytes / compressedBytes) : Math.round(1411 / bw);
  const origStr         = origBytes ? formatBytes(origBytes) : (dur + 's uncompressed WAV');
  const compStr         = formatBytes(compressedBytes);
  const cdRatio         = Math.round(1411 / bw);

  const p1 = 'System Alpha compressed your ' + dur + '-second clip from <strong>' + origStr +
    '</strong> down to roughly <strong>' + compStr + '</strong> of compressed data — a ' +
    '<strong>' + ratio + '&times; reduction</strong>. ' +
    'For context, a CD stores audio at 1,411 kbps; this system used ' + bw + ' kbps, which is ' +
    cdRatio + '&times; smaller than CD quality. The AI took <strong>' + processingTimeMs +
    ' ms</strong> to process ' + dur + ' seconds — ' +
    (dur > 0 ? Math.round(dur * 1000 / processingTimeMs) + '&times; faster than real-time' : 'very fast') + '.';

  // ── 2. Quality assessment — MOS-primary, SNR-fallback ───────────────────
  const mosVal  = parseFloat(m.mos_estimate) || 0;
  const hasMosN = m.mos_estimate != null && !isNaN(mosVal) && mosVal > 0;

  let qLevel, qColour, qDetail, qListen;
  if (hasMosN) {
    // Primary: perceptual quality (MOS estimate from MCD)
    if (mosVal >= 3.8) {
      qLevel  = 'Excellent';
      qColour = 'text-success';
      qDetail = 'Estimated MOS ' + mosVal.toFixed(1) + ' / 5 — near-transparent quality. ' +
                'The perceptual difference from the original is negligible.';
      qListen = 'Most listeners would not detect any meaningful difference, even on careful listening.';
    } else if (mosVal >= 3.0) {
      qLevel  = 'Good';
      qColour = 'text-success';
      qDetail = 'Estimated MOS ' + mosVal.toFixed(1) + ' / 5 — good perceptual quality. ' +
                'Minor spectral differences may exist but are unlikely to be bothersome.';
      qListen = 'Any differences are likely subtle — a slight change in high-frequency texture ' +
                'or brightness. Casual listening would not reveal a problem.';
    } else if (mosVal >= 2.0) {
      qLevel  = 'Fair';
      qColour = 'text-warning';
      qDetail = 'Estimated MOS ' + mosVal.toFixed(1) + ' / 5 — fair perceptual quality. ' +
                'Some spectral differences are measurable.';
      qListen = 'Tonal differences may be noticeable on attentive listening, particularly in ' +
                'high-frequency content like cymbals or consonants.';
    } else {
      qLevel  = 'Limited';
      qColour = 'text-danger';
      qDetail = 'Estimated MOS ' + mosVal.toFixed(1) + ' / 5. Significant perceptual differences detected.';
      qListen = 'Audible differences are likely. Using a lossless source (WAV/FLAC) typically ' +
                'produces better results than re-encoding a compressed format.';
    }
  } else {
    // Fallback: SNR with neural-codec appropriate thresholds
    // (Neural codecs optimise perception, not waveform accuracy — 15–25 dB SNR is normal
    //  and perceptually excellent at 24 kbps)
    if (snr >= 20) {
      qLevel  = 'Excellent';
      qColour = 'text-success';
      qDetail = 'SNR of ' + snr.toFixed(1) + ' dB — high-fidelity reconstruction.';
      qListen = 'Most listeners would not detect any meaningful difference.';
    } else if (snr >= 12) {
      qLevel  = 'Good';
      qColour = 'text-success';
      qDetail = 'SNR of ' + snr.toFixed(1) + ' dB — good reconstruction, typical for neural codecs at 24 kbps. ' +
                'Neural codecs prioritise perceptual quality over raw waveform accuracy.';
      qListen = 'Any differences are likely subtle. The audio may sound better than this number alone implies.';
    } else if (snr >= 7) {
      qLevel  = 'Good';
      qColour = 'text-success';
      qDetail = 'SNR of ' + snr.toFixed(1) + ' dB. Note: neural codecs routinely achieve this SNR range ' +
                'while maintaining near-transparent perceptual quality — waveform SNR underestimates ' +
                'how good the audio sounds.';
      qListen = 'Listen to the clips directly — the audio is likely to sound better than this metric suggests.';
    } else {
      qLevel  = 'Fair';
      qColour = 'text-warning';
      qDetail = 'SNR of ' + snr.toFixed(1) + ' dB. If your source was already compressed (MP3/AAC), ' +
                'low SNR is expected because waveform comparison detects differences between two ' +
                'different lossy codecs — it does not measure perceived quality.';
      qListen = 'Listen to the clips directly for a fair assessment. Consider using a lossless ' +
                'source (WAV/FLAC) for the most accurate quality metrics.';
    }
  }

  const p2 = 'Reconstruction quality: <strong class="' + qColour + '">' + qLevel + '</strong>. ' +
    qDetail + ' ' + qListen;

  // ── 3. What was discarded ────────────────────────────────────────────────
  const isHighQuality = (hasMosN && mosVal >= 3.0) || (!hasMosN && snr >= 12);
  const residualNote = isHighQuality
    ? 'The residual audio (what was discarded) should be very quiet — mostly imperceptible ' +
      'micro-details. Play it above to confirm.'
    : 'The residual audio (what was discarded) may contain perceptible content — ' +
      'listen to it above to hear exactly what the AI removed.';

  // ── 4. How the AI represented it ────────────────────────────────────────
  const frameMs = dur > 0 && frames > 0 ? (dur * 1000 / frames).toFixed(0) : 13;
  const cbDetail = 'The AI split your clip into <strong>' + frames + ' time frames</strong> ' +
    'of ~' + frameMs + ' ms each, using <strong>' + cb + ' codebook layers</strong> per frame. ' +
    'Layer 1 encodes coarse structure (the overall "shape" of the sound); each successive ' +
    'layer adds finer detail. The charts below show exactly which codes were chosen at each moment.';

  return [
    '<p class="mb-2">' + p1 + '</p>',
    '<p class="mb-2">' + p2 + '</p>',
    '<p class="mb-2">' + residualNote + '</p>',
    '<p class="mb-0">' + cbDetail + '</p>',
  ].join('');
}

/* ── State ─────────────────────────────────────────────────────────────────── */
let selectedSampleId  = null;
let selectedSampleUrl = null;
let uploadedFile      = null;
let audioProcessed    = false;
let answeredTrust     = {};
let mosAnswered       = false;
let currentJobId      = null;
let pollTimer         = null;

/* ── Sample selection ──────────────────────────────────────────────────────── */
function selectSample(tile) {
  document.querySelectorAll('.sample-tile').forEach(t => t.classList.remove('selected'));
  tile.classList.add('selected');
  selectedSampleId  = tile.dataset.sampleId;
  selectedSampleUrl = tile.dataset.sampleUrl;
  uploadedFile      = null;
  const upInput = document.getElementById('upload-input');
  if (upInput) upInput.value = '';
  const orig = document.getElementById('original-player');
  if (orig) { orig.src = selectedSampleUrl; }
  document.getElementById('original-preview').classList.remove('d-none');
  enableProcessButton(true);
  resetResults();
}

function onFileSelected(input) {
  if (!input.files || !input.files[0]) return;
  uploadedFile      = input.files[0];
  selectedSampleId  = null;
  selectedSampleUrl = null;
  document.querySelectorAll('.sample-tile').forEach(t => t.classList.remove('selected'));
  const orig = document.getElementById('original-player');
  if (orig) orig.src = URL.createObjectURL(uploadedFile);
  document.getElementById('original-preview').classList.remove('d-none');
  enableProcessButton(true);
  resetResults();
}

function enableProcessButton(state) {
  const btn = document.getElementById('btn-process');
  if (btn) btn.disabled = !state;
}

/* ── Processing ─────────────────────────────────────────────────────────────── */
function startProcessing() {
  if (!selectedSampleId && !uploadedFile) return;
  const cfg = window.SURVEY_CONFIG || {};
  const fd  = new FormData();
  fd.append('step', cfg.step || 0);
  if (uploadedFile) {
    fd.append('source', 'upload');
    fd.append('audio_file', uploadedFile);
  } else {
    fd.append('source', 'sample');
    fd.append('sample_id', selectedSampleId);
  }
  document.getElementById('card-processing').classList.remove('d-none');
  document.getElementById('btn-process').disabled = true;

  fetch('/api/process', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      if (data.error) { showProcessingError(data.error); return; }
      currentJobId = data.job_id;
      pollJobStatus(currentJobId);
    })
    .catch(err => showProcessingError('Network error: ' + err.message));
}

function pollJobStatus(jobId) {
  pollTimer = setInterval(() => {
    fetch(`/api/status/${jobId}`)
      .then(r => r.json())
      .then(job => {
        if (job.status === 'complete') {
          clearInterval(pollTimer);
          onProcessingComplete(job);
        } else if (job.status === 'error') {
          clearInterval(pollTimer);
          showProcessingError(job.error || 'Processing failed.');
        }
      })
      .catch(() => {});
  }, 800);
}

function onProcessingComplete(result) {
  document.getElementById('card-processing').classList.add('d-none');
  const cfg   = window.SURVEY_CONFIG || {};
  const codec = result.codec || cfg.codec;

  // ── Trim notification ────────────────────────────────────────────────────
  const trimNotif  = document.getElementById('trim-notification');
  const trimDetail = document.getElementById('trim-detail');
  if (trimNotif) {
    if (result.was_trimmed && result.original_duration_s) {
      const origS   = Math.round(result.original_duration_s);
      const origMin = Math.floor(origS / 60);
      const origSec = origS % 60;
      const origStr = origMin > 0
        ? origMin + 'm ' + origSec + 's'
        : origSec + 's';
      if (trimDetail) {
        trimDetail.textContent =
          'Your clip was ' + origStr + ' long — only the first 60 seconds were used. ';
      }
      trimNotif.classList.remove('d-none');
    } else {
      trimNotif.classList.add('d-none');
    }
  }

  // Set hidden clip ID
  const clipInput = document.getElementById('hidden-clip-id');
  if (clipInput) {
    clipInput.value = selectedSampleId || (uploadedFile ? uploadedFile.name : '');
  }

  // ── Current system framing ────────────────────────────────────────────────
  const badgeEl    = document.getElementById('system-type-badge');
  const noteEl     = document.getElementById('what-happened-codec-note');

  if (codec === 'A') {
    if (badgeEl) {
      badgeEl.textContent  = 'You are using: System Alpha - Process Transparency Shown';
      badgeEl.className    = 'badge px-3 py-1 bg-success';
      badgeEl.style.fontSize = '0.9rem';
    }
    if (noteEl) {
      noteEl.innerHTML =
        'This condition shows <strong>process transparency</strong> during ' +
        'compression. It provides information about what changed during ' +
        'compression, including quality estimates, residual audio, ' +
        'visualizations, and a plain-language summary. Use that information, ' +
        'along with what you hear, when rating this system.';
    }
  } else {
    if (badgeEl) {
      badgeEl.textContent  = 'You are using: System Beta - No Process Transparency';
      badgeEl.className    = 'badge px-3 py-1 bg-secondary';
      badgeEl.style.fontSize = '0.9rem';
    }
    if (noteEl) {
      noteEl.innerHTML =
        'This condition returns the processed audio without process ' +
        'transparency information. Use the original and processed clips, ' +
        'along with the information shown in this session, when rating this ' +
        'system.';
    }
  }

  // ── File size comparison ─────────────────────────────────────────────────
  const origBytes = result.original_size_bytes;
  // Prefer metrics values (Codec A) then top-level values (Codec B) then defaults
  const dur = result.metrics ? (parseFloat(result.metrics.duration_s)     || 30)
                             : (parseFloat(result.duration_s)              || 30);
  const bw  = result.metrics ? (parseFloat(result.metrics.bandwidth_kbps) || 6)
                             : (parseFloat(result.bandwidth_kbps)          || 6);
  const compBytes = Math.round(bw * 1000 / 8 * dur);
  const ratio     = origBytes ? Math.round(origBytes / compBytes) : Math.round(1411 / bw);

  const sizeRow  = document.getElementById('file-size-row');
  const sizeText = document.getElementById('file-size-text');
  if (sizeRow && sizeText) {
    const origStr = origBytes ? formatBytes(origBytes) : (dur + 's WAV');
    sizeText.innerHTML =
      'Original file: <strong>' + origStr + '</strong>' +
      ' &nbsp;&rarr;&nbsp; Compressed bitstream: <strong>' + formatBytes(compBytes) + '</strong>' +
      ' &nbsp;<span class="badge bg-accent ms-1">' + ratio + '&times; smaller</span>';
    sizeRow.classList.remove('d-none');
  }

  // ── Original audio ────────────────────────────────────────────────────────
  const origSrc = (document.getElementById('original-player') || {}).src || '';
  const p2 = document.getElementById('player-original-2');
  if (p2 && origSrc) p2.src = origSrc;

  // ── Processed audio ───────────────────────────────────────────────────────
  const pProc = document.getElementById('player-processed');
  if (pProc && result.processed_url) pProc.src = result.processed_url;

  // ── Timing badge ─────────────────────────────────────────────────────────
  const timingBadge = document.getElementById('timing-badge');
  if (timingBadge && result.processing_time_ms) {
    timingBadge.textContent = result.processing_time_ms + ' ms';
  }

  // ── Codec A: full explainability ─────────────────────────────────────────
  if (codec === 'A') {
    if (result.residual_url) {
      document.getElementById('player-residual').src = result.residual_url;
      document.getElementById('residual-row').classList.remove('d-none');
    }
    // Always refresh verdict/metrics — never retain stale data from a previous clip.
    // If no per-clip metrics are available, build a minimal fallback so duration
    // and file-size display correctly and the SNR-based verdict path fires cleanly
    // instead of showing numbers from a previous run.
    const metricsToRender = result.metrics || {
      snr_db: null, si_sdr_db: null, mcd_db: null, mos_estimate: null,
      bandwidth_kbps: bw, n_codebooks: 16, n_frames: 0,
      duration_s: dur, encode_time_s: 0
    };
    renderMetrics(metricsToRender);
    populateVerdict(metricsToRender, result.processing_time_ms, origBytes);

    // Plain-language narrative
    const sumEl = document.getElementById('summary-text');
    if (sumEl) {
      sumEl.innerHTML = generateNarrative(result.metrics, result.processing_time_ms, origBytes);
    }

    setFig('fig-waveforms',       result.fig_waveforms);
    setFig('fig-spectrograms',    result.fig_spectrograms);
    setFig('fig-heatmap',         result.fig_heatmap);
    setFig('fig-stats',           result.fig_stats);
    setFig('fig-input-analysis',  result.fig_input_analysis);
    setFig('fig-saliency',        result.fig_saliency);

    // ── Pre-compression input analysis panel ─────────────────────────────────
    const ia = result.input_analysis || {};
    if (ia.difficulty_label) {
      const diffColour = {
        'Easy':      'text-success',
        'Moderate':  'text-warning',
        'Hard':      'text-danger',
        'Very Hard': 'text-danger fw-bold',
      }[ia.difficulty_label] || 'text-muted';

      const badgeEl = document.getElementById('difficulty-badge');
      if (badgeEl) {
        badgeEl.textContent = ia.difficulty_label;
        badgeEl.className   = 'fs-4 fw-bold ' + diffColour;
      }
      const scoreEl = document.getElementById('difficulty-score');
      if (scoreEl) scoreEl.textContent = ia.difficulty_score + ' / 10';

      const predEl = document.getElementById('predicted-quality');
      if (predEl) predEl.textContent = ia.predicted_quality || '—';

      const maskedEl = document.getElementById('masked-energy');
      if (maskedEl) {
        const pct = ia.maskable_energy_frac != null
          ? Math.round(ia.maskable_energy_frac * 100) + '%'
          : '—';
        maskedEl.textContent = pct;
      }

      const labelInline = document.getElementById('difficulty-label-inline');
      if (labelInline) labelInline.textContent = ia.difficulty_label;

      const reasonsList = document.getElementById('difficulty-reasons');
      if (reasonsList && Array.isArray(ia.difficulty_reasons)) {
        reasonsList.innerHTML = ia.difficulty_reasons
          .map(r => `<li class="mb-1">${r}</li>`)
          .join('');
      }

      const listenEl = document.getElementById('listen-for-text');
      if (listenEl && Array.isArray(ia.listen_for) && ia.listen_for.length) {
        listenEl.textContent = ia.listen_for.join('; ') + '.';
      }
    }

    document.getElementById('explainability-panel').classList.remove('d-none');

  } else {
    // Codec B: show the absence panel so the lack of information is felt
    const absenceCard = document.getElementById('beta-absence-card');
    if (absenceCard) absenceCard.classList.remove('d-none');
  }

  document.getElementById('card-results').classList.remove('d-none');
  document.getElementById('card-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  audioProcessed = true;
  checkSubmitReady();
}

function showProcessingError(msg) {
  clearInterval(pollTimer);
  document.getElementById('card-processing').classList.add('d-none');
  enableProcessButton(true);
  const alert = document.createElement('div');
  alert.className = 'alert alert-danger alert-dismissible mt-3';
  alert.innerHTML = '<strong>Error:</strong> ' + msg +
    '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
  document.getElementById('card-source').querySelector('.card-body').appendChild(alert);
}

function resetResults() {
  audioProcessed = false;
  document.getElementById('card-results').classList.add('d-none');
  document.getElementById('card-processing').classList.add('d-none');
  const ep = document.getElementById('explainability-panel');
  if (ep) ep.classList.add('d-none');
  const rr = document.getElementById('residual-row');
  if (rr) rr.classList.add('d-none');
  const tn = document.getElementById('trim-notification');
  if (tn) tn.classList.add('d-none');
  checkSubmitReady();
}

/* ── Quality verdict ────────────────────────────────────────────────────────── */
function populateVerdict(m, processingTimeMs, origBytes) {
  const snr  = parseFloat(m.snr_db)         || 0;
  const mcd  = parseFloat(m.mcd_db)         || 0;
  const mos  = parseFloat(m.mos_estimate)   || 0;
  const bw   = parseFloat(m.bandwidth_kbps) || 24;
  const dur  = parseFloat(m.duration_s)     || 30;
  const hasMcd = m.mcd_db != null && !isNaN(parseFloat(m.mcd_db)) && mos > 0;

  // Verdict tier — MOS-primary when available; SNR-fallback with neural-codec thresholds
  let label, desc, cls, earCheck;
  if (hasMcd) {
    if (mos >= 3.8) {
      label    = 'Excellent';
      desc     = 'Near-transparent quality — the reconstruction is perceptually very close to the original.';
      cls      = 'verdict-excellent';
      earCheck = 'Most listeners would not detect any meaningful difference, even on careful repeated listening.';
    } else if (mos >= 3.0) {
      label    = 'Good';
      desc     = 'Good perceptual quality — any differences are minor and unlikely to be bothersome.';
      cls      = 'verdict-good';
      earCheck = 'If you can hear a difference, it is likely subtle — a slight change in high-frequency texture or brightness. Most casual listening would not reveal it.';
    } else if (mos >= 2.0) {
      label    = 'Fair';
      desc     = 'Fair quality — some perceptible differences in spectral character.';
      cls      = 'verdict-moderate';
      earCheck = 'Tonal differences may be noticeable on attentive listening, particularly in high-frequency content.';
    } else {
      label    = 'Limited';
      desc     = 'Significant spectral differences detected. Consider using a lossless 48kHz stereo source (WAV/FLAC) for best results.';
      cls      = 'verdict-limited';
      earCheck = 'Audible differences are likely. Lossless 48kHz stereo input produces much better results than upsampled mono or pre-compressed formats.';
    }
  } else {
    // SNR-based with neural-codec appropriate thresholds
    if (snr >= 20) {
      label    = 'Excellent';
      desc     = 'High-fidelity reconstruction — the audio is very well preserved.';
      cls      = 'verdict-excellent';
      earCheck = 'Most listeners would not detect any meaningful difference.';
    } else if (snr >= 12) {
      label    = 'Good';
      desc     = 'Good reconstruction quality, typical for neural codec compression at 24 kbps.';
      cls      = 'verdict-good';
      earCheck = 'If you can hear a difference, it is likely subtle. Neural codecs prioritise perception over raw waveform accuracy.';
    } else if (snr >= 7) {
      label    = 'Good';
      desc     = 'Perceptually good quality — neural codecs routinely achieve this SNR range while sounding near-transparent.';
      cls      = 'verdict-good';
      earCheck = 'Listen directly to the clips — the audio likely sounds better than this waveform metric suggests.';
    } else {
      label    = 'Fair';
      desc     = 'Some differences present. If input was a pre-compressed format (MP3/AAC), low SNR reflects waveform mismatch between two different lossy codecs.';
      cls      = 'verdict-moderate';
      earCheck = 'Listen directly to judge quality. For the most accurate metrics, use a lossless source (WAV or FLAC).';
    }
  }

  // Compression context
  const compBytes  = Math.round(bw * 1000 / 8 * dur);
  const ratio      = origBytes ? Math.round(origBytes / compBytes) : Math.round(1411 / bw);
  const origStr    = origBytes ? formatBytes(origBytes) : (dur + 's uncompressed WAV');

  // Perceptual metric row (MCD / MOS) if available — mcd/mos/hasMcd declared above
  const mcdRow = hasMcd
    ? '<div class="verdict-detail-row"><span class="verdict-detail-label">Perceptual quality</span>' +
      '<span>MCD ' + mcd.toFixed(1) + ' dB &bull; MOS est. ' + mos.toFixed(1) + ' / 5 — ' +
      (mcd < 10 ? 'excellent' : mcd < 20 ? 'good' : mcd < 30 ? 'fair' : 'limited') + '</span></div>'
    : '';

  // Supporting detail lines — primary card shows only perceptual quality + context.
  // Raw SNR / SI-SDR numbers are in the Technical Details section (renderMetrics grid).
  const detailHTML =
    mcdRow +
    '<div class="verdict-detail-row"><span class="verdict-detail-label">Compression</span>' +
    '<span>' + origStr + ' &rarr; ~' + formatBytes(compBytes) + ' (' + ratio + '&times; smaller)</span></div>' +
    '<div class="verdict-detail-row"><span class="verdict-detail-label">Processing time</span>' +
    '<span>' + processingTimeMs + ' ms for ' + dur + 's of audio</span></div>';

  // Populate DOM
  const circle = document.getElementById('verdict-circle');
  const lbl    = document.getElementById('verdict-label');
  const descEl = document.getElementById('verdict-description');
  const detail = document.getElementById('verdict-detail');
  const ear    = document.getElementById('verdict-ear-check');

  if (circle) { circle.className = 'verdict-circle mx-auto ' + cls; }
  if (lbl)    { lbl.textContent = label; }
  if (descEl) { descEl.textContent = desc; }
  if (detail) { detail.innerHTML = detailHTML; }
  if (ear)    { ear.textContent = earCheck; }
}

/* ── Metrics rendering ──────────────────────────────────────────────────────── */
function snrQuality(db) {
  // Neural-codec appropriate thresholds — these codecs optimise perception, not SNR
  if (db >= 20)  return { cls: 'mq-excellent', label: 'Excellent' };
  if (db >= 12)  return { cls: 'mq-good',      label: 'Good' };
  if (db >= 7)   return { cls: 'mq-good',      label: 'Good' };
  return           { cls: 'mq-fair',           label: 'Fair' };
}

function mcdQuality(db) {
  // Neural-codec calibrated thresholds — traditional speech-codec values
  // (<3 dB = excellent) do not apply; generative models inherently produce
  // higher MCD even when audio sounds perceptually good.
  if (db < 10)  return { cls: 'mq-excellent', label: 'Excellent' };
  if (db < 20)  return { cls: 'mq-good',      label: 'Good' };
  if (db < 30)  return { cls: 'mq-fair',       label: 'Fair' };
  return          { cls: 'mq-poor',            label: 'Limited' };
}

function mosQuality(s) {
  if (s >= 3.8) return { cls: 'mq-excellent', label: 'Excellent' };
  if (s >= 3.0) return { cls: 'mq-good',      label: 'Good' };
  if (s >= 2.0) return { cls: 'mq-fair',       label: 'Fair' };
  return          { cls: 'mq-poor',            label: 'Limited' };
}

function renderMetrics(m) {
  const grid = document.getElementById('metrics-grid');
  if (!grid) return;

  const snr  = parseFloat(m.snr_db)        || 0;
  const sidr = parseFloat(m.si_sdr_db)     || 0;
  const mcd  = parseFloat(m.mcd_db)        || 0;
  const mos  = parseFloat(m.mos_estimate)  || 0;
  const bw   = parseFloat(m.bandwidth_kbps) || 24;
  const hasMcd = m.mcd_db != null && !isNaN(parseFloat(m.mcd_db));

  const snrQ = snrQuality(snr);
  const sdrQ = snrQuality(sidr);

  // CD = 1411 kbps; calculate compression ratio
  const cdKbps      = 1411;
  const compression = Math.round(cdKbps / bw);

  const items = [
    {
      label:   'SNR',
      value:   snr.toFixed(1) + ' dB',
      explain: 'Signal-to-Noise Ratio — sample-level waveform fidelity.',
      quality: snrQ,
      tip:     'Higher is better. For neural codecs, 15–25 dB is typical for high quality — ' +
               'they optimise perception, not raw waveform accuracy.',
    },
    {
      label:   'MCD',
      value:   hasMcd ? mcd.toFixed(1) + ' dB' : '—',
      explain: 'Mel-Cepstral Distortion — perceptual quality in the frequency domain.',
      quality: hasMcd ? mcdQuality(mcd) : null,
      tip:     'Lower is better. Below 10 dB = excellent; 10–20 dB = good; 20–30 dB = fair. ' +
               'Neural-codec calibrated — traditional thresholds (<3 dB) do not apply here.',
    },
    {
      label:   'MOS Est.',
      value:   hasMcd && mos > 0 ? mos.toFixed(1) + ' / 5' : '—',
      explain: 'Estimated Mean Opinion Score — approximate perceived quality (1–5 scale).',
      quality: hasMcd && mos > 0 ? mosQuality(mos) : null,
      tip:     '5 = excellent; 4 = good; 3 = fair; 2 = poor. Derived from MCD, ' +
               'calibrated against neural codec listening tests.',
    },
    {
      label:   'SI-SDR',
      value:   sidr.toFixed(1) + ' dB',
      explain: 'Scale-Invariant SDR — quality measure that ignores volume differences.',
      quality: sdrQ,
      tip:     'Higher is better. Values above 20 dB are considered high quality.',
    },
    {
      label:   'Bitrate',
      value:   bw + ' kbps',
      explain: 'How much data the compressed audio uses per second.',
      quality: null,
      tip:     'A CD uses 1,411 kbps. This system used ' + bw + ' kbps — that is ' +
               compression + 'x smaller than CD quality.',
    },
    {
      label:   'AI Layers',
      value:   m.n_codebooks,
      explain: 'Number of codebook layers the AI used to represent your audio.',
      quality: null,
      tip:     'Each layer adds a finer level of detail — like rough sketch then fine details.',
    },
    {
      label:   'Time Frames',
      value:   m.n_frames,
      explain: 'The audio was split into this many small chunks to be processed.',
      quality: null,
      tip:     'Each frame is about 13 ms of audio.',
    },
    {
      label:   'Duration',
      value:   m.duration_s + ' s',
      explain: 'Length of the audio clip that was processed.',
      quality: null,
      tip:     '',
    },
  ];

  grid.innerHTML = items.map(it => `
    <div class="col-6 col-md-4 col-lg-3">
      <div class="metric-card text-center h-100" title="${it.tip}">
        <div class="metric-value">${it.value}</div>
        <div class="metric-label">${it.label}</div>
        <div class="metric-explain">${it.explain}</div>
        ${it.quality
          ? `<span class="metric-quality ${it.quality.cls}">${it.quality.label}</span>`
          : ''}
      </div>
    </div>
  `).join('');
}

function setFig(id, url) {
  const el = document.getElementById(id);
  if (el && url) el.src = url;
}

/* ── Lightbox ───────────────────────────────────────────────────────────────── */
function openLightbox(imgSrc, caption) {
  if (!imgSrc) return;
  const overlay = document.getElementById('lightbox-overlay');
  const img     = document.getElementById('lightbox-img');
  const cap     = document.getElementById('lightbox-caption');
  img.src       = imgSrc;
  cap.textContent = caption || '';
  overlay.classList.remove('d-none');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  document.getElementById('lightbox-overlay').classList.add('d-none');
  document.body.style.overflow = '';
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});

/* ── Trust scale ────────────────────────────────────────────────────────────── */
function onTrustAnswer(itemId, radioEl) {
  answeredTrust[itemId] = radioEl.value;
  const container = document.getElementById('ti-' + itemId);
  if (container) container.classList.add('answered');
  updateTrustCounter();
  checkSubmitReady();
}

function updateTrustCounter() {
  const cfg       = window.SURVEY_CONFIG || {};
  const total     = cfg.totalTrust || 12;
  const answered  = Object.keys(answeredTrust).length;
  const pct       = Math.round(answered / total * 100);

  const counter   = document.getElementById('trust-counter');
  const fill      = document.getElementById('counter-bar-fill');
  const text      = document.getElementById('counter-text');
  if (!counter) return;

  if (fill)  fill.style.width = pct + '%';
  if (text)  text.textContent = answered + ' of ' + total + ' answered';

  if (answered >= total) {
    counter.classList.add('complete');
  } else {
    counter.classList.remove('complete');
  }
}

/* ── MOS ────────────────────────────────────────────────────────────────────── */
function onMosAnswer(radioEl) {
  mosAnswered = true;
  checkSubmitReady();
}

/* ── Submit readiness ───────────────────────────────────────────────────────── */
function checkSubmitReady() {
  const cfg        = window.SURVEY_CONFIG || {};
  const totalTrust = cfg.totalTrust || 12;
  const trustOk    = Object.keys(answeredTrust).length >= totalTrust;
  const ready      = audioProcessed && trustOk && mosAnswered;

  const btn  = document.getElementById('btn-submit');
  const hint = document.getElementById('submit-hint');
  if (btn) btn.disabled = !ready;

  if (hint) {
    if (!audioProcessed) {
      hint.textContent = 'Compress an audio clip first to unlock the questionnaire.';
    } else if (!trustOk) {
      const rem = totalTrust - Object.keys(answeredTrust).length;
      hint.textContent = rem + ' trust question' + (rem !== 1 ? 's' : '') + ' still to answer.';
    } else if (!mosAnswered) {
      hint.textContent = 'Please rate the audio quality above.';
    } else {
      hint.textContent = '';
    }
  }
}

/* ── Form validation ────────────────────────────────────────────────────────── */
function showValidationError(msg) {
  const alert = document.getElementById('validation-alert');
  const text  = document.getElementById('validation-alert-text');
  if (!alert) return;
  if (text) text.textContent = msg;
  alert.classList.add('show');
  alert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  setTimeout(() => alert.classList.remove('show'), 5000);
}

function validateForm() {
  if (!audioProcessed) {
    showValidationError('Please compress an audio clip first — click the "Compress" button above.');
    return false;
  }
  const cfg = window.SURVEY_CONFIG || {};
  if (Object.keys(answeredTrust).length < (cfg.totalTrust || 12)) {
    const rem = (cfg.totalTrust || 12) - Object.keys(answeredTrust).length;
    showValidationError(
      rem + ' trust question' + (rem !== 1 ? 's' : '') +
      ' still unanswered. Scroll up to complete them.'
    );
    return false;
  }
  if (!mosAnswered) {
    showValidationError('Please rate the audio quality using the scale above before submitting.');
    return false;
  }
  return true;
}

/* ── Init ───────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  checkSubmitReady();
  // Prevent lightbox image click from closing the lightbox
  const lbImg = document.getElementById('lightbox-img');
  if (lbImg) lbImg.addEventListener('click', e => e.stopPropagation());
});
