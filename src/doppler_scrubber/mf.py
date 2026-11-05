"""Matched-filter search for Doppler curves too faint to show as per-frame peaks.

This is the attacker's view of a recording: brightness integrated along a
whole candidate curve, which pulls out tones that never clear a per-frame
threshold. The scrubber must find everything an attacker could, so it runs
the same kind of search. For each window, every (v, d, t_cpa) on a grid
and every rest frequency is scored by the ridge contrast along the curve
(on-curve brightness minus the two side bands at +-4 %), weighted towards
the sweeping part; the best curves are then polished with Nelder-Mead.

The search runs twice: on single lines, and on the mean contrast of the
first few harmonics. Doppler scales every harmonic by the same factor, so
summing them is where an attacker gains most: a stack whose lines are each
too faint to clear the single-line threshold can stand out as a whole.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize

from .tracks import C_SOUND, DopplerTrack, doppler_ratio

CONTRAST = 0.04
SWEEP_FLOOR = 0.15


def _sampler(W, t, f):
    """at(tq, fq): W at the nearest (frame, bin), 0 off the grid or where fq is not finite."""
    dt, df = t[1] - t[0], f[1] - f[0]
    F, T = W.shape
    # one zero row and column past the end: every miss reads 0 through a single flat take
    Wp = np.zeros((F + 1, T + 1), np.float32)
    Wp[:F, :T] = W
    flat = Wp.ravel()

    def at(tq, fq):
        ti, fi = np.broadcast_arrays(np.rint((tq - t[0]) / dt), np.rint((fq - f[0]) / df))
        ti = np.where((ti >= 0) & (ti < T), ti, T).astype(np.int64)
        fi = np.where((fi >= 0) & (fi < F), fi, F).astype(np.int64)
        return flat.take(fi * (T + 1) + ti)
    return at


def _contrast(at, tq, fq):
    return at(tq, fq) - 0.5 * (at(tq, fq * (1 + CONTRAST)) + at(tq, fq * (1 - CONTRAST)))


def _weights(R, tq):
    g = np.abs(np.gradient(R, tq, axis=-1))
    return g / np.maximum(g.max(axis=-1, keepdims=True), 1e-12) + SWEEP_FLOOR


def contrast_map(W, f):
    """On-bin brightness minus the mean of the bins at f*(1+-CONTRAST), for every (bin, frame):
    one lookup per curve point in the grid search instead of three."""
    df = f[1] - f[0]
    idx = np.arange(len(f))
    up = np.clip(np.rint((f * (1 + CONTRAST) - f[0]) / df).astype(int), 0, len(f) - 1)
    dn = np.clip(np.rint((f * (1 - CONTRAST) - f[0]) / df).astype(int), 0, len(f) - 1)
    C = W - 0.5 * (W[up] + W[dn])
    C[(f * (1 + CONTRAST) > f[-1])] = 0.0        # side band off the grid: no contrast known
    return C[idx]


def harmonic_map(C, f, harmonics):
    """Mean of C over the harmonics h*f, h = 1..harmonics, for each fundamental bin f
    (harmonics beyond the grid count as zero evidence)."""
    df = f[1] - f[0]
    out = np.zeros_like(C)
    for h in range(1, harmonics + 1):
        j = np.rint((h * f - f[0]) / df).astype(int)
        ok = j < len(f)
        out[ok] += C[j[ok]]
    return out / harmonics


def _window(at, ta, tb, f_lo, f_hi, v_grid, d_grid, f0_step, t0_step, n_curve, c, keep=8):
    """Top `keep` distinct grid curves in one window, as [(score, (f0, v, d, t_cpa))].

    Several are kept because on a coarse grid the maximum over thousands of noise curves
    can outscore a faint real curve; polishing then separates them (a real curve gains,
    a noise curve loses). `at` samples the (widened) contrast map."""
    tq = np.linspace(ta, tb, n_curve)
    t0s = np.arange(ta, tb, t0_step)
    f0s = np.arange(f_lo, f_hi, f0_step)
    cand = []
    for v in v_grid:
        for d in d_grid:
            R = doppler_ratio(tq[None, :], v, d, t0s[:, None], c)          # (T0, N)
            w = _weights(R, tq)
            w /= w.sum(axis=1, keepdims=True)
            Fq = f0s[None, None, :] * R[:, :, None]                       # (T0, N, F0)
            Tq = np.broadcast_to(tq[None, :, None], Fq.shape)
            s = np.einsum("tn,tnf->tf", w, at(Tq, Fq))
            for k in np.argsort(s, axis=None)[-keep:]:
                i, j = np.unravel_index(int(k), s.shape)
                cand.append((float(s[i, j]), (float(f0s[j]), float(v), float(d), float(t0s[i]))))
    cand.sort(key=lambda x: -x[0])
    out = []
    for sc, p in cand:     # distinct: another CPA time, or another frequency
        if all(abs(p[3] - q[3]) > 20 or abs(p[0] - q[0]) > 0.05 * p[0] for _, q in out):
            out.append((sc, p))
        if len(out) >= keep:
            break
    return out


def _polish(at, p, ta, tb, c, harmonics=1):
    tq = np.linspace(ta, tb, 400)

    def neg(q):
        f0, v, d, tc = q
        if not (20 < v < 250 and 60 < d < 30000 and f0 > 0):
            return 1e3
        R = doppler_ratio(tq, v, d, tc, c)
        w = _weights(R, tq)
        con = sum(_contrast(at, tq, h * f0 * R) for h in range(1, harmonics + 1)) / harmonics
        return -float(np.sum(w * con) / np.sum(w))

    f0, v, d, tc = p
    r = minimize(neg, p, method="Nelder-Mead",
                 options={"xatol": 1e-2, "fatol": 1e-4, "maxiter": 600,
                          "initial_simplex": np.array([p, [f0 + 2, v, d, tc], [f0, v + 10, d, tc],
                                                       [f0, v, d * 1.3, tc], [f0, v, d, tc + 2]])})
    return -float(r.fun), r.x


def matched_tracks(W, t, f, *, f_lo=40.0, f_hi=450.0, window_s=240.0, step_s=120.0,
                   min_score=2.0, max_score=4.0, null_quantile=0.9, min_sep_s=60.0, workers=4,
                   searches=(1, 4), harm_f_max=1000.0, c=C_SOUND) -> tuple[list[DopplerTrack], dict]:
    """Doppler curves whose polished matched-filter score clears this recording's own null.

    W is a whitened spectrogram (F, T). Returns (tracks, threshold): the threshold is the
    `null_quantile` of the scores the same search reaches on the time-reversed
    spectrogram, kept within [min_score, max_score] (dB). The cap matters: in a recording
    with strong steady tones a reversed curve resting on one scores high, and an uncapped
    null would then spare real flyovers. Returned tracks have no measured
    ridge (pts_t is None): their curve is the fitted model.

    `searches` lists the harmonic counts searched: 1 is single lines, n > 1 the mean
    contrast of harmonics 1..n, for fundamentals up to min(f[-1], harm_f_max)/n, each with its own null;
    its bounds are scaled by n**-0.4: averaging shrinks the noise spread by about 1/sqrt(n),
    but the best of many noise curves then sits relatively higher (in pure noise the
    4-harmonic search peaks at ~0.95 dB, the single-line one at ~2.1 dB).
    Returns (tracks, {harmonics searched: threshold}).
    """
    v_grid = np.arange(40, 175, 15.0)
    d_grid = np.geomspace(300, 12000, 10)
    starts = np.arange(t[0], max(t[-1] - window_s, t[0]) + 1e-9, step_s)

    def search(Wd, nh, lo_score, hi_f):
        # a grid curve 1-2 Hz off a ridge still scores on the widened map
        Cd = contrast_map(maximum_filter1d(Wd, 5, axis=0), f)
        if nh > 1:
            Cd = harmonic_map(Cd, f, nh)
        at_c, at_w = _sampler(Cd, t, f), _sampler(Wd, t, f)

        def job(ta):
            tb = min(ta + window_s, t[-1])
            return _window(at_c, ta, tb, f_lo, hi_f, v_grid, d_grid, 2.0, 3.0, 64, c)

        def polish(p):
            tc_heard = p[3] + p[2] / c
            lo, hi = max(tc_heard - 120, t[0]), min(tc_heard + 120, t[-1])
            return _polish(at_w, p, lo, hi, c, nh) + (lo, hi)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cands = [p for win in pool.map(job, starts) for sc, p in win if sc >= 0.6 * lo_score]
            return list(pool.map(polish, cands))

    found, thrs = [], {}
    for nh in searches:
        scale = nh ** -0.4
        # summed harmonics stay within harm_f_max: a wider fundamental range only adds noise
        # curves, which lift the null and its threshold and so spare faint real passes
        hi_f = min(f_hi, (f[-1] if nh == 1 else min(f[-1], harm_f_max)) / nh)
        if hi_f <= f_lo:
            continue
        polished = search(W, nh, min_score * scale, hi_f)
        # null: the same search on the time-reversed spectrogram, where a real pass would sweep
        # *up*, which no aircraft does; its scores are what this recording's clutter reaches
        null = [sc for sc, *_ in search(np.ascontiguousarray(W[:, ::-1]), nh, min_score * scale, hi_f)]
        thr = float(np.clip(np.quantile(null, null_quantile), min_score * scale, max_score * scale))             if null else min_score * scale
        thrs[nh] = thr
        # scores compared across the two searches in units of their own threshold
        found += [(sc / thr, sc, q, lo, hi) for sc, q, lo, hi in polished if sc >= thr]
    out = []
    for _, sc, q, lo, hi in sorted(found, key=lambda x: -x[0]):
        f0, v, d, tc = map(float, q)
        if any(abs((tc + d / c) - (k.t_cpa + k.d / c)) < min_sep_s and abs(f0 - k.f0) < 0.05 * f0 for k in out):
            continue
        R = doppler_ratio(np.array([lo, hi]), v, d, tc, c)
        out.append(DopplerTrack(lo, hi, f0, v, d, tc, rms=1.0, sweep=float(f0 * (R[0] - R[1])), snr_db=sc))
    return sorted(out, key=lambda k: k.t0), thrs
