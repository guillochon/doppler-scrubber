"""Find tone tracks that have the shape of an aircraft flyover.

A propeller (or rotor, or engine) tone heard from the ground falls in
frequency as the aircraft passes; for a straight pass it follows

    f(t) = f0 * c / (c + dr/dt),   r = sqrt(d^2 + v^2 (t_e - t_cpa)^2)

with propagation delay t = t_e + r/c. We pick narrowband peaks out of a
whitened spectrogram, link them into tracks, and keep the tracks that this
model fits to within a fraction of a hertz. Speech and music harmonics do
not survive that test, so they are left alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as nd
import scipy.signal as ss
from scipy.optimize import least_squares

C_SOUND = 340.0
V_MIN, D_MIN = 20.0, 60.0     # slower or closer than this is not a flyover
GRID_REJECT: float | None = 1.5   # Hz; candidates whose best coarse-grid fit is worse are not polished


def doppler_ratio(t, v, d, t_cpa, c=C_SOUND):
    """f/f0 heard at times t for a straight constant-speed pass (t_cpa = emission time).

    Exact, closed form, and broadcasts over v, d, t_cpa. With u = t_e - t_cpa and
    tau = t - t_cpa, the heard time obeys c(tau - u) = sqrt(d^2 + v^2 u^2), i.e.
    (c^2 - v^2) u^2 - 2 c^2 tau u + (c^2 tau^2 - d^2) = 0; the physical root is u < tau.
    """
    tau = np.asarray(t, float) - t_cpa
    k = c * c - v * v
    u = (c * c * tau - np.sqrt(c * c * v * v * tau * tau + k * d * d)) / k
    rr = v * v * u / np.sqrt(d * d + v * v * u * u)
    return c / (c + rr)


def ideal_curve(t, f0, v, d, t_cpa, c=C_SOUND):
    """Heard frequency at times t for a straight constant-speed pass (t_cpa = emission time)."""
    return f0 * doppler_ratio(t, v, d, t_cpa, c)


@dataclass
class DopplerTrack:
    t0: float          # first and last time the tone was seen, s
    t1: float
    f0: float          # rest frequency of the tracked line, Hz
    v: float           # speed, m/s
    d: float           # closest-approach slant distance, m
    t_cpa: float       # emission time of closest approach, s
    rms: float         # fit residual, Hz
    sweep: float       # frequency drop over the observed span, Hz
    snr_db: float      # mean whitened level along the track
    pts_t: np.ndarray | None = field(default=None, repr=False)   # measured ridge, if refined
    pts_f: np.ndarray | None = field(default=None, repr=False)

    def model(self, t):
        return ideal_curve(t, self.f0, self.v, self.d, self.t_cpa)

    def curve(self, t):
        """The measured ridge where it was seen, the model (joined to it) beyond."""
        t = np.asarray(t, float)
        m = self.model(t)
        if self.pts_t is None or len(self.pts_t) < 2:
            return m
        res = self.pts_f - self.model(self.pts_t)
        ok = np.isfinite(res)
        if ok.sum() < 2:
            return m
        pt, pr = self.pts_t[ok], res[ok]
        # residual is interpolated inside the measured span and held at its end values outside
        return m + np.interp(t, pt, pr)

    def to_dict(self):
        return {k: float(getattr(self, k)) for k in
                ("t0", "t1", "f0", "v", "d", "t_cpa", "rms", "sweep", "snr_db")}


def whiten(L: np.ndarray, smooth_frames: int = 3) -> np.ndarray:
    """dB above the per-bin median, lightly smoothed in time."""
    W = L - np.median(L, axis=1, keepdims=True)
    if smooth_frames > 1:
        W = nd.uniform_filter1d(W, smooth_frames, axis=1)
    return W


def _link(W, t, f, thresh_db, prominence_db, max_slope, max_gap_s):
    dt = t[1] - t[0]
    df = f[1] - f[0]
    max_gap = max(1, int(round(max_gap_s / dt)))
    act_t, act_f, act_a, act_last = [], [], [], []
    done = []
    for j in range(W.shape[1]):
        col = W[:, j]
        pk, _ = ss.find_peaks(col, height=thresh_db, prominence=prominence_db)
        if len(pk):
            a, b, c = col[np.maximum(pk - 1, 0)], col[pk], col[np.minimum(pk + 1, len(col) - 1)]
            den = a - 2 * b + c
            off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
            fp = f[pk] + df * np.clip(off, -0.5, 0.5)
            amp = b
        else:
            fp = amp = np.zeros(0)
        used = np.zeros(len(fp), bool)
        if act_t and len(fp):
            pred = np.empty(len(act_t))
            gap = np.empty(len(act_t))
            for i in range(len(act_t)):
                tt, ff = act_t[i], act_f[i]
                k = min(len(tt), 8)
                slope = (ff[-1] - ff[-k]) / (tt[-1] - tt[-k]) if k > 1 else 0.0
                slope = float(np.clip(slope, -max_slope, max_slope))
                pred[i] = ff[-1] + slope * (t[j] - tt[-1])
                gap[i] = j - act_last[i]
            # a young track has no reliable slope yet, so it gets the full sweep allowance;
            # an established one must land near its own extrapolation
            young = np.array([len(a) < 4 for a in act_t])
            tol = 1.2 * df + np.where(young, max_slope * dt, 0.4) * gap
            D = np.abs(fp[None, :] - pred[:, None])
            ii, kk = np.nonzero(D < tol[:, None])
            order = np.argsort(D[ii, kk])
            taken = set()
            for o in order:
                i, k = ii[o], kk[o]
                if i in taken or used[k]:
                    continue
                act_t[i].append(t[j]); act_f[i].append(fp[k]); act_a[i].append(amp[k]); act_last[i] = j
                used[k] = True; taken.add(i)
        for k in np.flatnonzero(~used):
            act_t.append([t[j]]); act_f.append([fp[k]]); act_a.append([amp[k]]); act_last.append(j)
        keep = [i for i in range(len(act_t)) if j - act_last[i] <= max_gap]
        for i in range(len(act_t)):
            if j - act_last[i] > max_gap:
                done.append((act_t[i], act_f[i], act_a[i]))
        act_t = [act_t[i] for i in keep]; act_f = [act_f[i] for i in keep]
        act_a = [act_a[i] for i in keep]; act_last = [act_last[i] for i in keep]
    done += list(zip(act_t, act_f, act_a))
    return [(np.array(a), np.array(b), np.array(c)) for a, b, c in done]


_GRID_V = np.array([25.0, 35.0, 50.0, 70.0, 100.0, 140.0, 200.0])
_GRID_D = np.geomspace(80.0, 20000.0, 14)


def _grid_fit(tt, ff, c, n_tc=40):
    """Vectorised coarse fit: every (v, d, t_cpa) on a grid, best f0 in closed form.
    Returns the top candidates as (params, rms), best first."""
    tc = np.linspace(tt[0] - 300.0, tt[-1] + 300.0, n_tc)
    V, D, TC = np.meshgrid(_GRID_V, _GRID_D, tc, indexing="ij")
    V, D, TC = V.ravel()[:, None], D.ravel()[:, None], TC.ravel()[:, None]
    S = doppler_ratio(tt[None, :], V, D, TC, c)                 # (combos, points)
    f0 = (S @ ff) / np.einsum("ij,ij->i", S, S)
    rms = np.sqrt(np.mean((f0[:, None] * S - ff) ** 2, axis=1))
    order = np.argsort(rms)[:3]
    return [(np.array([f0[i], V[i, 0], D[i, 0], TC[i, 0]]), float(rms[i])) for i in order]


def fit_track(tt, ff, c=C_SOUND, reject_rms: float | None = None):
    """Straight-pass fit: vectorised grid, then least-squares polish; returns (params, rms).

    If even the best grid point is worse than `reject_rms`, it is returned unpolished:
    the caller is going to reject the track anyway.
    """
    lo = [ff.min() * 0.6, V_MIN, D_MIN, tt[0] - 900.0]
    hi = [ff.max() * 1.6, 250.0, 30000.0, tt[-1] + 900.0]

    def res_on(ts, fs):
        def res(p):
            m = ideal_curve(ts, *p, c=c)
            return np.where(np.isfinite(m), m - fs, 50.0)
        return res

    def solve(res, p0, nfev):
        try:
            return least_squares(res, p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.5, max_nfev=nfev).x
        except ValueError:
            return None

    # stage 1: vectorised grid on at most 80 points; stage 2: polish the best few on all points
    sub = np.linspace(0, len(tt) - 1, min(len(tt), 80)).astype(int)
    starts = _grid_fit(tt[sub], ff[sub], c)
    if reject_rms is not None and starts[0][1] > reject_rms:
        return starts[0]
    res = res_on(tt, ff)
    best = None
    for p0, _ in starts:
        p = solve(res, np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6), 60)
        if p is None:
            continue
        rms = float(np.sqrt(np.mean(res(p) ** 2)))
        if best is None or rms < best[1]:
            best = (p, rms)
    return best


def find_tracks(L: np.ndarray, t: np.ndarray, f: np.ndarray, *, thresh_db: float = 5.0,
                prominence_db: float = 4.0, max_slope: float = 12.0, max_gap_s: float = 2.0,
                min_dur_s: float = 12.0, max_rms: float = 0.3, min_sweep: float = 1.0,
                max_jitter: float = 0.12) -> list[DopplerTrack]:
    """Doppler-shaped tracks in a dB spectrogram L (F, T)."""
    W = whiten(L)
    out = []
    for tt, ff, aa in _link(W, t, f, thresh_db, prominence_db, max_slope, max_gap_s):
        if len(tt) < 8 or tt[-1] - tt[0] < min_dur_s:
            continue
        # cheap jitter test before the expensive fit: a real tone is smooth frame to frame
        if np.median(np.abs(np.diff(ff, 2))) > max_jitter:
            continue
        # a Doppler curve is close to a low-order polynomial over a track's span (a quintic once
        # the span is long enough to include the turn at closest approach); noise chains are not
        s = (tt - tt.mean()) / max(np.ptp(tt), 1e-9)
        deg = 3 if np.ptp(tt) < 40 else 5
        if np.sqrt(np.mean((np.polyval(np.polyfit(s, ff, deg), s) - ff) ** 2)) > max_rms:
            continue
        fit = fit_track(tt, ff, reject_rms=GRID_REJECT)
        if fit is None:
            continue
        p, rms = fit
        model = ideal_curve(tt, *p)
        sweep = float(model[0] - model[-1]) if np.all(np.isfinite(model[[0, -1]])) else 0.0
        if rms > max_rms or sweep < min_sweep:
            continue
        tr = DopplerTrack(float(tt[0]), float(tt[-1]), *map(float, p), rms, sweep, float(aa.mean()), tt, ff)
        tr = refine(W, t, f, tr, max_rms=max_rms)
        # pinned at a fit bound: not a flyover (birdsong and other tonal sources end up here)
        if tr.v <= V_MIN * 1.05 or tr.v >= 245.0 or tr.d <= D_MIN * 1.05 or tr.d >= 29000.0:
            continue
        out.append(tr)
    return _dedupe(out)


def refine(W, t, f, tr: DopplerTrack, *, extend_s: float = 150.0, search_hz: float = 0.75,
           keep_db: float = 4.0, window_s: float = 8.0, iters: int = 2, max_rms: float = 0.3) -> DopplerTrack:
    """Follow the ridge outward from the track along its own model, then refit.

    Joins fragments of one pass (broken at the fast sweep near closest approach,
    or where the tone crosses another line) and follows faint tails, so the
    corridor later follows the measured ridge rather than an extrapolation.
    """
    df, dt = f[1] - f[0], t[1] - t[0]
    nwin = max(2, int(round(window_s / dt)))
    half = max(1, int(round(search_hz / df)))
    for _ in range(iters):
        js = np.flatnonzero((t >= tr.t0 - extend_s) & (t <= tr.t1 + extend_s))
        if len(js) < 3:
            break
        model = tr.model(t[js])
        mid = int(np.argmin(np.abs(t[js] - 0.5 * (tr.t0 + tr.t1))))
        pts: dict[int, float] = {}
        for step in (1, -1):
            off, vals, run = 0.0, [], []
            for i in range(mid, len(js) if step > 0 else -1, step):
                if not np.isfinite(model[i]):
                    break
                c = (model[i] + off - f[0]) / df
                lo, hi = int(round(c)) - half, int(round(c)) + half
                if lo < 1 or hi >= len(f) - 1:
                    break
                col = W[lo:hi + 1, js[i]]
                k = int(np.argmax(col))
                val = float(col[k])
                vals.append(val)
                if len(vals) > nwin and np.mean(vals[-nwin:]) < keep_db:
                    break
                if val >= keep_db:
                    b = lo + k
                    a, bb, cc = W[b - 1, js[i]], W[b, js[i]], W[b + 1, js[i]]
                    den = a - 2 * bb + cc
                    fp = f[b] + df * (float(np.clip(0.5 * (a - cc) / den, -0.5, 0.5)) if den != 0 else 0.0)
                    pts[i] = fp
                    off = 0.7 * off + 0.3 * (fp - model[i])
                    run.append(i)
            # drop the tail of the final window that let the walk stop on noise
            for i in run[-nwin:]:
                if W[int(round((pts[i] - f[0]) / df)), js[i]] < keep_db + 1.0:
                    pts.pop(i, None)
        if len(pts) < 8:
            break
        ii = np.array(sorted(pts))
        ts, fs = t[js[ii]], np.array([pts[i] for i in ii])
        fit = fit_track(ts, fs)
        if fit is None or fit[1] > 2 * max_rms:
            break
        p, rms = fit
        m = ideal_curve(ts, *p)
        sweep = float(m[0] - m[-1]) if np.all(np.isfinite(m[[0, -1]])) else tr.sweep
        lvl = float(np.mean([W[int(round((q - f[0]) / df)), j] for q, j in zip(fs, js[ii])]))
        tr = DopplerTrack(float(ts[0]), float(ts[-1]), *map(float, p), rms, sweep, lvl, ts, fs)
    return tr


def _dedupe(tracks: list[DopplerTrack]) -> list[DopplerTrack]:
    """Drop tracks that duplicate a stronger one (same curve, overlapping time)."""
    tracks = sorted(tracks, key=lambda r: -(r.t1 - r.t0))
    kept: list[DopplerTrack] = []
    for tr in tracks:
        tt = np.linspace(tr.t0, tr.t1, 50)
        dup = False
        for k in kept:
            if tr.t1 < k.t0 or tr.t0 > k.t1:
                continue
            a, b = tr.curve(tt), k.curve(tt)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 10 and np.median(np.abs(a[ok] - b[ok])) < 1.0:
                dup = True
                break
        if not dup:
            kept.append(tr)
    return sorted(kept, key=lambda r: r.t0)
