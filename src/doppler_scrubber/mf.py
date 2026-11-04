"""Matched-filter search for Doppler curves too faint to show as per-frame peaks.

This is the attacker's view of a recording: brightness integrated along a
whole candidate curve, which pulls out tones that never clear a per-frame
threshold. The scrubber must find everything an attacker could, so it runs
the same kind of search. For each window, every (v, d, t_cpa) on a grid
and every rest frequency is scored by the ridge contrast along the curve
(on-curve brightness minus the two side bands at +-4 %), weighted towards
the sweeping part; the best curves are then polished with Nelder-Mead.
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
    dt, df = t[1] - t[0], f[1] - f[0]
    F, T = W.shape

    def at(tq, fq):
        ti = np.rint((tq - t[0]) / dt).astype(np.int64)
        fi = np.rint((fq - f[0]) / df).astype(np.int64)
        ok = (ti >= 0) & (ti < T) & (fi >= 0) & (fi < F) & np.isfinite(fq)
        out = np.zeros(np.broadcast(ti, fi).shape, np.float32)
        out[ok] = W[fi[ok], ti[ok]]
        return out
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


def _window(Cfat, t, f, ta, tb, f_lo, f_hi, v_grid, d_grid, f0_step, t0_step, n_curve, c, keep=8):
    """Top `keep` distinct grid curves in one window, as [(score, (f0, v, d, t_cpa))].

    Several are kept because on a coarse grid the maximum over thousands of noise curves
    can outscore a faint real curve; polishing then separates them (a real curve gains,
    a noise curve loses)."""
    at = _sampler(Cfat, t, f)
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


def _polish(W, t, f, p, ta, tb, c):
    at = _sampler(W, t, f)
    tq = np.linspace(ta, tb, 400)

    def neg(q):
        f0, v, d, tc = q
        if not (20 < v < 250 and 60 < d < 30000 and f0 > 0):
            return 1e3
        R = doppler_ratio(tq, v, d, tc, c)
        w = _weights(R, tq)
        return -float(np.sum(w * _contrast(at, tq, f0 * R)) / np.sum(w))

    f0, v, d, tc = p
    r = minimize(neg, p, method="Nelder-Mead",
                 options={"xatol": 1e-2, "fatol": 1e-4, "maxiter": 600,
                          "initial_simplex": np.array([p, [f0 + 2, v, d, tc], [f0, v + 10, d, tc],
                                                       [f0, v, d * 1.3, tc], [f0, v, d, tc + 2]])})
    return -float(r.fun), r.x


def matched_tracks(W, t, f, *, f_lo=40.0, f_hi=450.0, window_s=240.0, step_s=120.0,
                   min_score=2.0, max_score=4.0, null_quantile=0.9, min_sep_s=60.0, workers=4,
                   c=C_SOUND) -> tuple[list[DopplerTrack], float]:
    """Doppler curves whose polished matched-filter score clears this recording's own null.

    W is a whitened spectrogram (F, T). Returns (tracks, threshold): the threshold is the
    `null_quantile` of the scores the same search reaches on the time-reversed
    spectrogram, kept within [min_score, max_score] (dB). The cap matters: in a recording
    with strong steady tones a reversed curve resting on one scores high, and an uncapped
    null would then spare real flyovers. Returned tracks have no measured
    ridge (pts_t is None): their curve is the fitted model.
    """
    v_grid = np.arange(40, 175, 15.0)
    d_grid = np.geomspace(300, 12000, 10)
    starts = np.arange(t[0], max(t[-1] - window_s, t[0]) + 1e-9, step_s)

    def search(Wd):
        # a grid curve 1-2 Hz off a ridge still scores on the widened map
        Cd = contrast_map(maximum_filter1d(Wd, 5, axis=0), f)

        def job(ta):
            tb = min(ta + window_s, t[-1])
            return _window(Cd, t, f, ta, tb, f_lo, f_hi, v_grid, d_grid, 2.0, 3.0, 64, c)

        def polish(p):
            tc_heard = p[3] + p[2] / c
            lo, hi = max(tc_heard - 120, t[0]), min(tc_heard + 120, t[-1])
            return _polish(Wd, t, f, p, lo, hi, c) + (lo, hi)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cands = [p for win in pool.map(job, starts) for sc, p in win if sc >= 0.6 * min_score]
            return list(pool.map(polish, cands))

    polished = search(W)
    # null: the same search on the time-reversed spectrogram, where a real pass would sweep
    # *up*, which no aircraft does; its scores are what this recording's clutter reaches
    null = [sc for sc, *_ in search(np.ascontiguousarray(W[:, ::-1]))]
    thr = float(np.clip(np.quantile(null, null_quantile), min_score, max_score)) if null else min_score
    out = []
    for sc, q, lo, hi in sorted(polished, key=lambda x: -x[0]):
        if sc < thr:
            continue
        f0, v, d, tc = map(float, q)
        if any(abs((tc + d / c) - (k.t_cpa + k.d / c)) < min_sep_s and abs(f0 - k.f0) < 0.05 * f0 for k in out):
            continue
        R = doppler_ratio(np.array([lo, hi]), v, d, tc, c)
        out.append(DopplerTrack(lo, hi, f0, v, d, tc, rms=1.0, sweep=float(f0 * (R[0] - R[1])), snr_db=sc))
    return sorted(out, key=lambda k: k.t0), thr
