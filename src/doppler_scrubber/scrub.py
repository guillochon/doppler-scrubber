"""Subtract Doppler tones and refill the gap with matching noise.

For every accepted track, each line (the tracked tone times a harmonic
ratio) is heterodyned to 0 Hz with its own tracked phase and low-passed,
which gives the tone's complex amplitude. Small frequency errors are
corrected from the drift of that amplitude's phase, the reconstructed
sinusoid is subtracted with a Wiener gain (so only what rises above the
local floor is removed), and the narrow band that was taken out is refilled
with independent noise of the floor's level. The corridor ends up neither
bright nor dark, and nothing more than ~1 Hz from a line is touched.

Detection and editing run on a 4 kHz copy of the audio; the change is
upsampled and added to the original full-rate samples, so everything above
about 1.9 kHz is bit-identical apart from the final re-encode.
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.ndimage as nd
import scipy.signal as ss

from . import stft
from .mf import _sampler, contrast_map, matched_tracks
from .tracks import DopplerTrack, find_tracks, whiten

LOW_SR = 4000
DUAL_MONO = 10 ** (-50 / 20)   # channels closer than this (rms, re the mix) are treated as one


@dataclass
class ScrubConfig:
    fmin: float = 30.0             # band searched and cleaned, Hz
    fmax: float = 1700.0           # the 4 kHz working copy is flat to here (decode + upsample lose
                                   # 0.2 dB at 1.7 kHz, 1.7 dB at 1.8, 6 dB at 1.9: no clean removal above)
    n_detect: int = 8192           # detection STFT length at 4 kHz (2.05 s, 0.49 Hz bins)
    hop_detect: int = 1024
    passes: int = 2                # detect -> clean -> detect again on the cleaned audio
    sub_harmonics: int = 2         # the tracked line may itself be the h-th harmonic, h <= this
    extend_s: float = 90.0         # clean this far beyond where the tone was seen
    lp_s: float = 1.0              # amplitude low-pass length (bandwidth ~ 1/lp_s Hz)
    noise_offsets: tuple[float, ...] = (-6.0, -4.0, 4.0, 6.0)   # where the floor is measured, Hz
    lp_model_s: float = 0.4        # low-pass for tracks known only from a model (~2.5 Hz band)
    line_ratio: float = 1.15       # a harmonic line is replaced only if its span-mean power exceeds this x floor
    line_db: float = 1.0           # ... and only if its 30 s ridge contrast reaches this somewhere (dB)
    steady_db: float = 8.0         # a line this far above its neighbours at the 20th percentile is protected
    mf_min_score: float | None = 2.0    # matched-filter floor (dB); the real threshold is set per recording
    mf_null_quantile: float = 0.9  # ... at this quantile of the time-reversed (up-sweep) null scores
    mf_max_score: float = 2.5      # ... but never above this: a blind search on Oyster Bay showed 2.5 suffices
    mf_harmonics: int = 4          # also search the mean of harmonics 1..N (0 or 1: single lines only)
    seed: int = 0
    workers: int = field(default_factory=lambda: max(1, min(8, (os.cpu_count() or 2) // 2)))
    thresh_db: float = 6.0
    max_rms: float = 0.3
    min_dur_s: float = 12.0


def _ratios(cfg: ScrubConfig) -> list[float]:
    return sorted({m / h for h in range(1, cfg.sub_harmonics + 1) for m in range(1, 64)})


def _lowpass(z: np.ndarray, sr: int, length_s: float) -> np.ndarray:
    n = max(3, int(length_s * sr) | 1)
    w = np.hanning(n + 2)[1:-1]
    return ss.oaconvolve(z, w / w.sum(), mode="same")


def _smooth(p: np.ndarray, sr: int, length_s: float) -> np.ndarray:
    return nd.uniform_filter1d(p, max(1, int(length_s * sr)), mode="nearest")


BB_RATE = 10        # Hz: everything after the heterodyne is ~1 Hz wide, so estimate at 10 Hz
MID_RATE = 200      # Hz: intermediate rate where the +-4..6 Hz side bands are shifted
BB_PASS = 3.0       # Hz: half-width kept flat in baseband (the widest band, lp_model_s, is ~+-2 Hz)


@lru_cache(maxsize=None)
def _fir(sr: float, dec: int, pass_hz: float, stop_hz: float) -> np.ndarray:
    """Kaiser low-pass, flat to pass_hz and 90 dB down from stop_hz, with (len-1)/2 a
    multiple of dec so that its delay is a whole number of output samples."""
    n, beta = ss.kaiserord(90.0, (stop_hz - pass_hz) / (sr / 2))
    half = -(-((n - 1) // 2 + 1) // dec) * dec
    return ss.firwin(2 * half + 1, (pass_hz + stop_hz) / 2, window=("kaiser", beta), fs=sr)


def _decimate(z: np.ndarray, sr: float, dec: int, pass_hz: float) -> np.ndarray:
    """Keep every `dec`-th sample after a low-pass that is flat to pass_hz and rejects
    everything that would alias to within pass_hz of 0; output sample i is at input sample
    i*dec. A boxcar is not enough here: its -13 dB sidelobes let loud sound hundreds of hertz
    away alias into the band being measured, inflating the side-band level and so the noise
    put back (heard as a whistle where the band was quiet)."""
    h = _fir(float(sr), dec, pass_hz, sr / dec - pass_hz)
    k = (len(h) - 1) // 2 // dec
    return ss.upfirdn(h, z, 1, dec)[k:k + -(-len(z) // dec)]


def _remove_line(x: np.ndarray, sr: int, finst: np.ndarray, cfg: ScrubConfig,
                 rng: np.random.Generator, lp_s: float, own: bool) -> np.ndarray:
    """Delta that replaces the narrow band following `finst` with fresh noise.

    The tone's complex amplitude is estimated in baseband and subtracted in full,
    and independent noise at the side-band level is put back, so the band holds
    noise of the right level and nothing else. `own` marks the line the track was
    detected on, which is always replaced; any other harmonic line is replaced only
    if, averaged over the whole span, it holds clearly more power than its side bands.
    """
    # two-stage decimation: 4 kHz -> MID_RATE once per iteration (the only full-rate work),
    # side-band offsets applied there, then -> BB_RATE for estimation
    dec1 = sr // MID_RATE
    mr = sr / dec1
    dec2 = int(round(mr / BB_RATE))
    br = mr / dec2                                  # baseband sample rate
    tt = np.arange(len(x)) / sr
    tm = np.arange(-(-len(x) // dec1)) * dec1 / sr
    tb = np.arange(-(-len(tm) // dec2)) * dec2 / mr
    mid_pass = max(abs(o) for o in cfg.noise_offsets) + BB_PASS   # the side bands, with their width
    finst = finst.copy()
    for it in range(3):
        phi = 2 * np.pi * np.cumsum(finst) / sr
        e = _unit(-phi)
        xm = _decimate(_times(x, e), sr, dec1, mid_pass)
        a = 2 * _lowpass(_decimate(xm, mr, dec2, BB_PASS), br, lp_s)
        side = np.sort([_smooth(np.abs(2 * _lowpass(_decimate(xm * np.exp(-2j * np.pi * o * tm), mr, dec2, BB_PASS),
                                                    br, lp_s)) ** 2, br, 10.0)
                        for o in cfg.noise_offsets], axis=0)
        # trimmed mean of the side bands: robust to another tone in one of them, and unbiased
        # for noise (a plain median of skewed power estimates reads low)
        pn = side[1:-1].mean(axis=0) if len(side) > 2 else side.mean(axis=0)
        pa = _smooth(np.abs(a) ** 2, br, lp_s)
        if it == 0 and not own and np.mean(np.abs(a) ** 2) < cfg.line_ratio * np.mean(pn):
            return np.zeros_like(x)                 # this harmonic carries nothing: leave it alone
        if it == 2:
            break
        # frequency error from the drift of the amplitude's phase, where the tone is clear
        strong = pa > 4 * pn
        if strong.sum() < br:
            break
        dphi = np.gradient(np.unwrap(np.angle(a))) * br / (2 * np.pi)
        w = strong.astype(float)
        ws = _smooth(w, br, 2.0)
        corr = np.clip(np.where(ws > 0.2, _smooth(dphi * w, br, 2.0) / np.maximum(ws, 1e-3), 0.0), -0.5, 0.5)
        finst += np.interp(tt, tb, corr)
    # never put back more than the band held: where the side bands read louder than the band
    # itself (another sound in them), refilling at their level would add a tone-like band
    pn = np.minimum(pn, _smooth(np.abs(a) ** 2, br, 10.0))
    # fade the replacement in and out over 5 s at the ends of the span, and over 2 s where
    # the line leaves fmin..fmax (a harmonic can sweep across fmax)
    n_r = max(1, min(int(5 * br), len(tb) // 2))
    g = np.ones(len(tb))
    g[:n_r] = g[-n_r:][::-1] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n_r) + 0.5) / n_r)
    fb = np.interp(tb, tt, finst)
    # (2*L - 1 of the smoothed in-range indicator: zero from the crossing on, not half-on there)
    g *= np.clip(2 * _lowpass(((fb >= cfg.fmin) & (fb <= cfg.fmax)).astype(float), br, 2.0) - 1, 0, 1)
    # the subtraction removes the band with the low-pass's response H; noise shaped by
    # sqrt(1 - (1-H)^2) puts back exactly what it took, so the band's spectrum stays flat at
    # the side-band density (a plain low-passed noise piles up at the centre: a bright core
    # with dark flanks, a curve again under fine frequency resolution)
    nt = len(tb)
    kern = np.zeros(nt)
    n_lp = max(3, int(lp_s * br) | 1)
    w = np.hanning(n_lp + 2)[1:-1]
    kern[np.arange(-(n_lp // 2), n_lp // 2 + 1) % nt] = w / w.sum()
    H = np.clip(np.real(np.fft.fft(kern)), 0, 1)
    wn = np.fft.ifft(np.fft.fft(rng.standard_normal(nt) + 1j * rng.standard_normal(nt)) *
                     np.sqrt(1 - (1 - H) ** 2)) / np.sqrt(2)
    # the side bands were measured through H: their power is the density times mean(H^2)
    fill = np.sqrt(np.clip(1 - (1 - g) ** 2, 0, 1) * pn / np.mean(H ** 2)) * wn
    d = -_carrier(g * a - fill, tt, tb, e)
    # closed loop: measure the edited band the way it was estimated and scale it onto pn
    # (2 s at a time). Whatever the estimate got wrong, a fill too loud (a whistle) or tone
    # left over (a bright trace), the band ends up at its neighbours' level.
    for _ in range(2):
        ao = 2 * _lowpass(_decimate(_decimate(_times(x + d, e), sr, dec1, mid_pass), mr, dec2, BB_PASS), br, lp_s)
        k = np.sqrt(np.clip(pn / np.maximum(_smooth(np.abs(ao) ** 2, br, 2.0), 1e-30), 0.25, 4.0))
        d += _carrier(g * (k - 1) * ao, tt, tb, e)
    return d


def _carrier(bb: np.ndarray, tt: np.ndarray, tb: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Real signal at the line's frequency carrying the baseband amplitude bb (sampled at tb):
    Re(bb / e), written out in real arithmetic (|e| = 1, so 1/e = conj(e))."""
    return np.interp(tt, tb, bb.real) * e.real + np.interp(tt, tb, bb.imag) * e.imag


def _unit(phi: np.ndarray) -> np.ndarray:
    """exp(1j * phi), from cos and sin written straight into one array (3x faster than np.exp)."""
    e = np.empty(len(phi), np.complex128)
    np.cos(phi, out=e.real)
    np.sin(phi, out=e.imag)
    return e


def _times(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """x * e for real x and complex e, without promoting x to complex first."""
    out = np.empty(len(x), np.complex128)
    np.multiply(x, e.real, out=out.real)
    np.multiply(x, e.imag, out=out.imag)
    return out


def _apply(x: np.ndarray, sr: int, tracks: list[DopplerTrack], cfg: ScrubConfig, stats: dict) -> np.ndarray:
    """Delta that removes `tracks` (and their harmonics) from mono signal x.

    Tracks run one after another, each on the previous one's residual (lines of
    different tracks can cross). The lines of one track are fixed multiples of
    the same curve and never cross, so they are estimated in parallel threads
    from the same input; numpy releases the GIL in the heavy array work.
    """
    y = x.astype(np.float64)
    lines = 0
    if not tracks:
        return y - x
    # the detection spectrogram's ridge-contrast map, used to skip empty harmonic lines
    # before any full-rate work (most of a track's lines carry nothing)
    ts, fsp, L = stft.power_db(x, sr, cfg.n_detect, cfg.hop_detect, cfg.fmin, cfg.fmax)
    cmap = _sampler(contrast_map(whiten(L), fsp), ts, fsp)
    nwin = max(1, int(round(30.0 / (ts[1] - ts[0]))))
    for ti, tr in enumerate(tracks):
        # a ridge track is extended past where it was seen; a matched-filter track's span
        # already covers +-2 min around its closest approach
        ext = cfg.extend_s if tr.pts_t is not None else 0.0
        a = max(0, int((tr.t0 - ext) * sr))
        b = min(len(x), int((tr.t1 + ext) * sr) + 1)
        if b - a < 4 * sr:
            continue
        tg = np.arange(a / sr, b / sr, 0.05)
        fg = tr.curve(tg)
        ok = np.isfinite(fg)
        if ok.sum() < 20:
            continue
        tg, fg = tg[ok], fg[ok]
        a, b = int(tg[0] * sr), min(len(x), int(tg[-1] * sr) + 1)
        fs = np.interp(np.arange(a, b) / sr, tg, fg)
        ratios = [r for r in _ratios(cfg) if not (fs.max() * r < cfg.fmin or fs.min() * r > cfg.fmax)]
        tq = ts[(ts >= a / sr) & (ts <= b / sr)]
        fq = np.interp(tq, tg, fg)

        def visible(r):
            # 30 s running mean of ridge contrast along the line reaches line_db somewhere
            cv = cmap(tq, fq * r)
            return len(cv) > 0 and np.max(nd.uniform_filter1d(cv, min(nwin, len(cv)))) >= cfg.line_db
        keep = [k for k, r in enumerate(ratios) if r == 1.0 or visible(r)]
        # a measured ridge is accurate to a fraction of a hertz; a matched-filter model
        # (no ridge) only to a hertz or so, so it gets a wider band
        lp = cfg.lp_s if tr.pts_t is not None else cfg.lp_model_s
        seg = y[a:b].copy()

        def one(k):
            # each line gets its own seeded stream, so results do not depend on thread timing
            return _remove_line(seg, sr, fs * ratios[k], cfg, np.random.default_rng([cfg.seed, ti, k]),
                                lp, own=ratios[k] == 1.0)
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            for d in pool.map(one, keep):
                y[a:b] += d
        lines += len(keep)
    stats["lines"] = stats.get("lines", 0) + lines
    return y - x


def scan(x: np.ndarray, sr: int, cfg: ScrubConfig, log=None) -> list[DopplerTrack]:
    """Ridge tracks (tones visible frame by frame) plus matched-filter tracks (tones only
    visible integrated along a whole curve, as an attacker would look for them)."""
    t, f, L = stft.power_db(x, sr, cfg.n_detect, cfg.hop_detect, cfg.fmin, cfg.fmax)
    tracks = find_tracks(L, t, f, thresh_db=cfg.thresh_db, max_rms=cfg.max_rms, min_dur_s=cfg.min_dur_s)
    if cfg.mf_min_score is not None:
        mt, thr = matched_tracks(whiten(L), t, f, f_lo=max(cfg.fmin, 40.0), f_hi=min(cfg.fmax, 450.0),
                                 min_score=cfg.mf_min_score, max_score=cfg.mf_max_score,
                                 null_quantile=cfg.mf_null_quantile,
                                 searches=(1, cfg.mf_harmonics) if cfg.mf_harmonics > 1 else (1,),
                                 workers=cfg.workers)
        if log:
            log("matched filter: thresholds " + ", ".join(f"{v:.2f} dB ({'1 line' if k == 1 else f'{k} harmonics'})"
                                                          for k, v in thr.items())
                + f" from the up-sweep null, {len(mt)} curves above them")
        tracks += [m for m in mt if not any(_same(m, k) for k in tracks)]
    return tracks


def _same(a: DopplerTrack, b: DopplerTrack) -> bool:
    if a.t1 < b.t0 or a.t0 > b.t1:
        return False
    tt = np.linspace(max(a.t0, b.t0), min(a.t1, b.t1), 30)
    da, db = a.curve(tt), b.curve(tt)
    ok = np.isfinite(da) & np.isfinite(db)
    return ok.sum() > 5 and np.median(np.abs(da[ok] - db[ok])) < 1.0


def steady_lines(x: np.ndarray, sr: int, cfg: ScrubConfig) -> np.ndarray:
    """Frequencies of lines present nearly all the time (mains hum, machinery): bins whose
    20th-percentile level stands well above that of the neighbouring bins."""
    t, f, L = stft.power_db(x, sr, 16384, 4096, cfg.fmin, cfg.fmax)
    if L.shape[1] < 10:
        return np.zeros(0)
    p20 = np.percentile(L, 20, axis=1)
    excess = p20 - nd.median_filter(p20, size=41, mode="nearest")
    pk, _ = ss.find_peaks(excess, height=cfg.steady_db, distance=4)
    return f[pk]


def _steady_part(x: np.ndarray, sr: int, freqs: np.ndarray) -> np.ndarray:
    """The steady lines' own signal: each heterodyned at its frequency and averaged over
    ~4 s (a +-0.25 Hz band, which follows mains drift but not a passing aircraft)."""
    dec = sr // BB_RATE
    br = sr / dec
    tt = np.arange(len(x)) / sr
    tb = np.arange(-(-len(x) // dec)) * dec / sr
    out = np.zeros(len(x))
    for fs in freqs:
        e = _unit(-2 * np.pi * fs * tt)
        a = 2 * _lowpass(_decimate(x * e, sr, dec, 1.0), br, 4.0)
        up = np.interp(tt, tb, a.real) + 1j * np.interp(tt, tb, a.imag)
        out += np.real(up / e)
    return out


def scrub_array(x: np.ndarray, sr: int = LOW_SR, cfg: ScrubConfig | None = None, log=None):
    """x: (channels, samples) at `sr`. Returns (delta, tracks, stats).

    Steady lines are taken out before anything else and are not part of what is
    scrubbed, so a flyover crossing mains hum leaves the hum as it was; the
    returned delta is then added to the original x, which still has them.
    """
    cfg = cfg or ScrubConfig()
    log = log or (lambda m: None)
    x = np.atleast_2d(np.asarray(x, np.float64))
    steady = steady_lines(x.mean(axis=0), sr, cfg)
    log(f"steady lines protected: {[round(float(v), 1) for v in steady]}")
    if len(steady):
        x = x - np.stack([_steady_part(ch, sr, steady) for ch in x])
    mono = x.mean(axis=0)
    tracks: list[DopplerTrack] = []
    cur = mono
    passes_run = 0
    for _ in range(cfg.passes):
        passes_run += 1
        new = [tr for tr in scan(cur, sr, cfg, log) if not any(_same(tr, k) for k in tracks)]
        if not new:
            break
        tracks += new
        log(f"pass {passes_run}: {len(new)} new tracks ({sum(t.pts_t is None for t in new)} matched-filter)")
        if passes_run < cfg.passes:                     # the cleaned mix only feeds the next pass
            cur = cur + _apply(cur, sr, new, cfg, {})   # earlier tracks are already gone from cur
    stats: dict = {}
    # channels that are copies of each other (dual mono, up to codec noise) get one shared
    # delta: half the work, and the edit stays mono instead of differing between the ears
    same = len(x) > 1 and np.sqrt(np.mean((x - mono) ** 2)) <= DUAL_MONO * np.sqrt(np.mean(mono ** 2))
    chans = [mono] if same else list(x)
    log(f"{len(tracks)} tracks; removing them from {len(chans)} channel(s)"
        + (f" (the {len(x)} channels are the same signal)" if same else ""))
    # channels in parallel, each with its own pool of line workers
    ch_stats = [{} for _ in chans]
    with ThreadPoolExecutor(max_workers=len(chans)) as pool:
        delta = np.stack(list(pool.map(lambda k: _apply(chans[k], sr, tracks, cfg, ch_stats[k]), range(len(chans)))))
    stats["lines"] = sum(st.get("lines", 0) for st in ch_stats)
    if same:
        delta = np.repeat(delta, len(x), axis=0)
    sig = float(np.sqrt(np.mean(x ** 2)) + 1e-20)
    stats.update(passes=passes_run, tracks=len(tracks), steady_lines_hz=[round(float(v), 2) for v in steady],
                 delta_rms_db_re_signal=float(20 * np.log10(np.sqrt(np.mean(delta ** 2)) / sig + 1e-20)))
    return delta, tracks, stats


# ---------------------------------------------------------------- file I/O

def _probe(path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=sample_rate,channels", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["sample_rate"]), int(s["channels"])


def _decode(path, sr: int, ch: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", str(ch),
                          "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).T.copy()


_AUDIO_ONLY = {".wav": ["-c:a", "pcm_s16le"], ".flac": ["-c:a", "flac"],
               ".m4a": ["-c:a", "aac", "-b:a", "256k"], ".opus": ["-c:a", "libopus", "-b:a", "192k"],
               ".mp3": ["-c:a", "libmp3lame", "-b:a", "256k"]}
_VIDEO = {".mp4": ["-c:a", "aac", "-b:a", "256k"], ".mov": ["-c:a", "aac", "-b:a", "256k"],
          ".webm": ["-c:a", "libopus", "-b:a", "192k"], ".mkv": ["-c:a", "flac"]}


def _writer(src, out: Path, sr: int, ch: int, keep_video: bool):
    ext = out.suffix.lower()
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", "-"]
    if ext in _VIDEO and keep_video:
        # picture only (V: no cover art), and none of the source's data streams (telemetry)
        cmd += ["-i", str(src), "-map", "1:V?", "-map", "0:a", "-c:v", "copy"] + _VIDEO[ext]
    elif ext in _AUDIO_ONLY:
        cmd += _AUDIO_ONLY[ext]
    elif ext in _VIDEO:
        cmd += _VIDEO[ext]
    else:
        raise ValueError(f"unsupported output extension {ext!r}")
    # no tags, chapters or encoder string: creation time, GPS and device place a recording too
    cmd += ["-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact"]
    return subprocess.Popen(cmd + [str(out)], stdin=subprocess.PIPE)


def scrub_file(src, out, cfg: ScrubConfig | None = None, report=None, keep_video: bool = True, log=None):
    """Scrub an audio or video file. Video streams are copied, audio is re-encoded once."""
    cfg = cfg or ScrubConfig()
    src, out = Path(src), Path(out)
    sr, ch = _probe(src)
    low = _decode(src, LOW_SR, ch)
    delta, tracks, stats = scrub_array(low, LOW_SR, cfg, log=log)

    frac = Fraction(sr, LOW_SR)
    up, down = frac.numerator, frac.denominator
    block_low = LOW_SR * 10
    block_hi = sr * 10
    pad = LOW_SR
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-ac", str(ch),
                           "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"], stdout=subprocess.PIPE)
    wr = _writer(src, out, sr, ch, keep_video)
    clipped = 0
    k = 0
    while True:
        buf = rd.stdout.read(block_hi * ch * 4)
        if not buf:
            break
        y = np.frombuffer(buf, np.float32).reshape(-1, ch).T.astype(np.float64)
        a = k * block_low
        s0 = max(0, a - pad)
        seg = delta[:, s0:a + block_low + pad]
        if seg.shape[1]:
            up_seg = ss.resample_poly(seg, up, down, axis=1)
            off = (a - s0) * sr // LOW_SR
            d = up_seg[:, off:off + y.shape[1]]
            y[:, :d.shape[1]] += d
        clipped += int(np.sum(np.abs(y) > 1.0))
        wr.stdin.write(np.clip(y, -1.0, 1.0).T.astype(np.float32).tobytes())
        k += 1
    rd.wait()
    wr.stdin.close()
    if wr.wait() != 0:
        raise RuntimeError("ffmpeg failed writing " + str(out))
    stats["clipped_samples"] = clipped
    result = {"input": str(src), "output": str(out), "sample_rate": sr, "channels": ch,
              "config": asdict(cfg), "stats": stats, "tracks": [t.to_dict() for t in tracks]}
    if report:
        Path(report).write_text(json.dumps(result, indent=2))
    return result
