"""Optional GPU backend (CuPy, CUDA) for the removal: all of one track's lines at once.

It follows scrub._remove_line step for step, with the lines of a track stacked along a
first axis: the full-rate mix-down and the final carrier are fused CUDA kernels, the
estimation at 200 Hz and 10 Hz runs batched, and the per-line branches of the CPU code (a
harmonic left alone, a frequency correction that stops early) become masks. The noise put
back is drawn on the CPU from the same seeded generators, so the result matches the CPU
path to rounding.

Install with the `gpu` extra (`uv sync --extra gpu`); `available()` says whether a usable
CUDA device was found.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np

_CP = None


@lru_cache(maxsize=1)
def available() -> str | None:
    """Name of the CUDA device CuPy can use, or None (no CuPy, no device, or it fails)."""
    global _CP
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                return None
            # a real round trip through a kernel, an FFT and a filter: catches a missing
            # runtime library or a driver too old for this CuPy build
            x = cp.arange(64, dtype=cp.float64)
            assert abs(float(cp.fft.ifft(cp.fft.fft(x)).real[5]) - 5.0) < 1e-9
            import cupyx.scipy.ndimage  # noqa: F401
            import cupyx.scipy.signal  # noqa: F401
            name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        _CP = cp
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return None


def use(device: str) -> bool:
    """Whether to run on the GPU for ScrubConfig.device ("auto", "gpu" or "cpu")."""
    if device == "cpu":
        return False
    if device == "gpu":
        if available() is None:
            raise RuntimeError("device='gpu' but no usable CUDA device (install the `gpu` extra: "
                               "uv sync --extra gpu)")
        return True
    if device == "auto":
        return available() is not None
    raise ValueError(f"device must be 'auto', 'gpu' or 'cpu', not {device!r}")


# ---------------------------------------------------------------- kernels

@lru_cache(maxsize=None)
def _kernels():
    cp = _CP
    mix = cp.ElementwiseKernel(
        "float64 x, float64 phi, float64 c", "complex128 xe",
        "double s, co; sincos(c * phi, &s, &co); xe = complex<double>(x * co, x * s);",
        "ds_mix_down")
    carrier = cp.ElementwiseKernel(
        "float64 phi, raw float64 r, raw complex128 bc, int32 nl, int32 nb, int32 step, float64 c",
        "float64 out",
        """
        long long k = i / step;
        double fr = (double)(i - k * step) / step;
        double acc = 0.0;
        for (int l = 0; l < nl; l++) {
            double re, im;
            if (k >= nb - 1) {
                re = bc[(long long)l * nb + nb - 1].real(); im = bc[(long long)l * nb + nb - 1].imag();
            } else {
                complex<double> b0 = bc[(long long)l * nb + k], b1 = bc[(long long)l * nb + k + 1];
                re = b0.real() + fr * (b1.real() - b0.real());
                im = b0.imag() + fr * (b1.imag() - b0.imag());
            }
            double s, co; sincos(c * r[l] * phi, &s, &co);
            acc += re * co + im * s;
        }
        out = -acc;
        """, "ds_carrier")
    return mix, carrier


# ---------------------------------------------------------------- batched helpers (last axis)

def _decimate(z, h, dec):
    from cupyx.scipy.signal import upfirdn
    k = (len(h) - 1) // 2 // dec
    n = -(-z.shape[-1] // dec)
    return upfirdn(h, z, 1, dec, axis=-1)[..., k:k + n]


@lru_cache(maxsize=None)
def _hann(n: int):
    w = np.hanning(n + 2)[1:-1]
    return _CP.asarray(w / w.sum())


def _lowpass(z, sr, length_s):
    from cupyx.scipy.ndimage import convolve1d
    w = _hann(max(3, int(length_s * sr) | 1))
    if z.dtype.kind == "c":
        return (convolve1d(z.real, w, axis=-1, mode="constant")
                + 1j * convolve1d(z.imag, w, axis=-1, mode="constant"))
    return convolve1d(z, w, axis=-1, mode="constant")


def _smooth(p, sr, length_s):
    from cupyx.scipy.ndimage import uniform_filter1d
    return uniform_filter1d(p, max(1, int(length_s * sr)), axis=-1, mode="nearest")


def _trimmed(side):
    return side[1:-1].mean(axis=0) if len(side) > 2 else side.mean(axis=0)


def _upsample_linear(b, step, n):
    """np.interp of b (sampled every `step` points) at points 0..n-1, held after the last."""
    cp = _CP
    i = cp.arange(n)
    k = i // step
    fr = (i - k * step) / step
    k0 = cp.minimum(k, b.shape[-1] - 1)
    k1 = cp.minimum(k + 1, b.shape[-1] - 1)
    return b[..., k0] + fr * (b[..., k1] - b[..., k0])


# ---------------------------------------------------------------- the removal

def remove_lines(x: np.ndarray, fs: np.ndarray, ratios, owns, rngs, sr: int, cfg, lp_s: float,
                 fir, rates) -> np.ndarray:
    """Summed delta of scrub._remove_line(x, sr, fs * r, cfg, rng, lp_s, own) over the lines.

    `fir` is scrub._fir and `rates` is (MID_RATE, BB_RATE, BB_PASS), passed in to keep the
    filters and rates the CPU path's own."""
    cp = _CP
    mid_rate, bb_rate, bb_pass = rates
    mix, carrier = _kernels()
    n, nl = len(x), len(ratios)
    r = cp.asarray(np.asarray(ratios, float))
    owns = cp.asarray(np.asarray(owns, bool))
    dec1 = sr // mid_rate
    mr = sr / dec1
    dec2 = int(round(mr / bb_rate))
    br = mr / dec2
    nm = -(-n // dec1)
    nb = -(-nm // dec2)
    tm = cp.arange(nm) * dec1 / sr
    mid_pass = max(abs(o) for o in cfg.noise_offsets) + bb_pass
    h1 = cp.asarray(fir(float(sr), dec1, mid_pass, sr / dec1 - mid_pass))
    h2 = cp.asarray(fir(float(mr), dec2, bb_pass, mr / dec2 - bb_pass))
    xg = cp.asarray(x, cp.float64)
    phi = cp.cumsum(cp.asarray(fs, cp.float64))     # the lines' phases are r * phi
    c = -2.0 * math.pi / sr

    # full rate -> MID_RATE, a few lines at a time (each needs a full-rate complex array)
    chunk = max(1, min(nl, (1 << 26) // max(n, 1)))
    xm0 = cp.empty((nl, nm), cp.complex128)
    for a in range(0, nl, chunk):
        rb = r[a:a + chunk, None]
        xm0[a:a + chunk] = _decimate(mix(xg[None, :], phi[None, :] * rb, c), h1, dec1)

    side_e = cp.stack([cp.exp(-2j * math.pi * o * tm) for o in cfg.noise_offsets])
    cf = cp.zeros((nl, nb))
    psi = cp.zeros((nl, nm))
    running = cp.ones(nl, bool)
    active = cp.ones(nl, bool)
    A = cp.zeros((nl, nb), cp.complex128)
    PN = cp.zeros((nl, nb))
    PSI = cp.zeros((nl, nm))
    CF = cp.zeros((nl, nb))
    for it in range(3):
        xm = xm0 if it == 0 else xm0 * cp.exp(-1j * psi)
        a_ = 2 * _lowpass(_decimate(xm, h2, dec2), br, lp_s)
        side_p = cp.abs(2 * _lowpass(_decimate(xm[None] * side_e[:, None], h2, dec2), br, lp_s)) ** 2
        pn = _trimmed(cp.sort(_smooth(side_p, br, 10.0), axis=0))
        pns = _trimmed(cp.sort(_smooth(side_p, br, cfg.fill_s), axis=0))
        pnf = cp.clip(pn, pns / cfg.fill_cap, pns * cfg.fill_cap)
        pa = _smooth(cp.abs(a_) ** 2, br, lp_s)
        upd = running[:, None]
        A = cp.where(upd, a_, A)
        PN = cp.where(upd, pnf, PN)
        PSI = cp.where(upd, psi, PSI)
        CF = cp.where(upd, cf, CF)
        if it == 0:
            # a harmonic line that carries nothing is left alone
            active = owns | (cp.mean(cp.abs(a_) ** 2, axis=1) >= cfg.line_ratio * cp.mean(pn, axis=1))
            running &= active
        if it == 2:
            break
        strong = pa > 4 * pn
        running &= strong.sum(axis=1) >= br
        if not bool(running.any()):
            break
        dphi = cp.gradient(cp.unwrap(cp.angle(a_), axis=1), axis=1) * br / (2 * math.pi)
        w = strong.astype(float)
        ws = _smooth(w, br, 2.0)
        corr = cp.clip(cp.where(ws > 0.2, _smooth(dphi * w, br, 2.0) / cp.maximum(ws, 1e-3), 0.0), -0.5, 0.5)
        cf = cf + corr * running[:, None]
        cfm = _upsample_linear(cf, dec2, nm)
        psi = 2 * math.pi * ((cp.cumsum(cfm, axis=1) - cfm) * dec1 + cfm) / sr

    keep = cp.flatnonzero(active)
    if len(keep) == 0:
        return np.zeros(n)
    keep_h = cp.asnumpy(keep)
    A, PN, PSI, CF, r = A[keep], PN[keep], PSI[keep], CF[keep], r[keep]
    nk = len(keep_h)

    PN = cp.minimum(PN, _smooth(cp.abs(A) ** 2, br, 10.0))     # PN holds the fill level (fill_s)
    n_r = max(1, min(int(5 * br), nb // 2))
    g = np.ones(nb)
    g[:n_r] = g[-n_r:][::-1] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n_r) + 0.5) / n_r)
    idx = cp.minimum(cp.arange(nb) * (dec1 * dec2), n - 1)
    fb = cp.asarray(fs, cp.float64)[idx][None, :] * r[:, None] + CF
    inr = ((fb >= cfg.fmin) & (fb <= cfg.fmax)).astype(float)
    g = cp.asarray(g)[None, :] * cp.clip(2 * _lowpass(inr, br, 2.0) - 1, 0, 1)
    kern = np.zeros(nb)
    n_lp = max(3, int(lp_s * br) | 1)
    wl = np.hanning(n_lp + 2)[1:-1]
    kern[np.arange(-(n_lp // 2), n_lp // 2 + 1) % nb] = wl / wl.sum()
    H = np.clip(np.real(np.fft.fft(kern)), 0, 1)
    # the same seeded noise as the CPU path: each kept line's generator, in its own order
    noise = np.stack([rngs[k].standard_normal(nb) + 1j * rngs[k].standard_normal(nb) for k in keep_h])
    wn = cp.fft.ifft(cp.fft.fft(cp.asarray(noise), axis=1) * cp.asarray(np.sqrt(1 - (1 - H) ** 2)),
                     axis=1) / math.sqrt(2)
    fill = cp.sqrt(cp.clip(1 - (1 - g) ** 2, 0, 1) * PN / float(np.mean(H ** 2))) * wn
    B = g * A - fill
    for _ in range(2):
        ao = A - _lowpass(_decimate(_upsample_linear(B, dec2, nm), h2, dec2), br, lp_s)
        k = cp.sqrt(cp.clip(PN / cp.maximum(_smooth(cp.abs(ao) ** 2, br, 2.0), 1e-30), 0.25, 4.0))
        B = B - g * (k - 1) * ao
    psib = PSI[:, cp.minimum(cp.arange(nb) * dec2, nm - 1)]
    Bc = cp.ascontiguousarray(B * cp.exp(1j * psib))
    out = carrier(phi, r, Bc, np.int32(nk), np.int32(nb), np.int32(dec1 * dec2), c)
    return cp.asnumpy(out)


# ---------------------------------------------------------------- matched-filter grid

_GRID_SRC = r"""
extern "C" __global__ void ds_grid(const float* C, int nf, int nt, const long long* ti,
                                   const double* R, const double* W, int ncurve, int n,
                                   const double* f0s, int nf0, double f_first, double df,
                                   double* out) {
    // one thread per (curve, f0): out = sum_n W[curve, n] * C[bin(f0s * R[curve, n]), ti[n]],
    // in the same order and rounding as mf._grid_scores (no fused multiply-adds)
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)ncurve * nf0) return;
    int cu = idx / nf0, j = idx % nf0;
    double acc = 0.0;
    for (int k = 0; k < n; k++) {
        long long tn = ti[k];
        if (tn < 0 || tn >= nt) continue;
        double r = R[(long long)cu * n + k];
        double fi = rint(__ddiv_rn(__dsub_rn(__dmul_rn(f0s[j], r), f_first), df));
        if (fi >= 0 && fi < nf)
            acc = __dadd_rn(acc, __dmul_rn(W[(long long)cu * n + k], (double)C[(long long)fi * nt + tn]));
    }
    out[idx] = acc;
}
"""


@lru_cache(maxsize=None)
def _grid_kernel():
    return _CP.RawKernel(_GRID_SRC, "ds_grid")


def to_device(a):
    return _CP.asarray(a)


def grid_scores(Cg, ti, R, W, f0s, f_first, df) -> np.ndarray:
    """mf._grid_scores for a stack of curve sets at once: R, W (..., N) -> scores (..., F0)."""
    cp = _CP
    shp = R.shape[:-1]
    n = R.shape[-1]
    Rg = cp.asarray(np.ascontiguousarray(R.reshape(-1, n)))
    Wg = cp.asarray(np.ascontiguousarray(W.reshape(-1, n)))
    ncurve, nf0 = Rg.shape[0], len(f0s)
    out = cp.empty(ncurve * nf0)
    total = ncurve * nf0
    _grid_kernel()(((total + 255) // 256,), (256,),
                   (Cg, np.int32(Cg.shape[0]), np.int32(Cg.shape[1]), cp.asarray(ti, cp.int64),
                    Rg, Wg, np.int32(ncurve), np.int32(n), cp.asarray(f0s, cp.float64),
                    np.int32(nf0), np.float64(f_first), np.float64(df), out))
    return cp.asnumpy(out).reshape(*shp, nf0)
