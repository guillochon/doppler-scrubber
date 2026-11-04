"""Chunked weighted-overlap-add STFT used both to look for tones and to edit them.

Edits are returned as a *delta* signal (ISTFT of Z_edited - Z), so bins that
were not edited contribute exactly nothing and the caller adds the delta to
the untouched original.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _frames(x: np.ndarray, n: int, hop: int):
    pad = np.concatenate([np.zeros(n, x.dtype), x, np.zeros(n, x.dtype)])
    view = np.lib.stride_tricks.sliding_window_view(pad, n)[::hop]
    return pad, view


def frame_times(n_samples: int, sr: int, n: int, hop: int) -> np.ndarray:
    """Centre time (s, in the unpadded signal) of every frame."""
    k = (n_samples + 2 * n - n) // hop + 1
    return (np.arange(k) * hop + n / 2 - n) / sr


def power_db(x: np.ndarray, sr: int, n: int, hop: int, fmin: float, fmax: float,
             chunk: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, f, L): band-limited STFT power in dB, shape (F, T)."""
    win = np.hanning(n + 1)[:-1].astype(np.float32)
    _, view = _frames(x.astype(np.float32), n, hop)
    f = np.fft.rfftfreq(n, 1 / sr)
    band = (f >= fmin) & (f <= fmax)
    out = []
    for j0 in range(0, len(view), chunk):
        Z = np.fft.rfft(view[j0:j0 + chunk] * win, axis=1)[:, band]
        out.append((10 * np.log10(np.abs(Z) ** 2 + 1e-20)).astype(np.float32))
    return frame_times(len(x), sr, n, hop), f[band], np.concatenate(out).T


def edit(x: np.ndarray, sr: int, n: int, hop: int,
         fn: Callable[[np.ndarray, np.ndarray, np.ndarray], None], chunk: int = 1024) -> np.ndarray:
    """Return the delta signal produced by `fn(Z, t, f)` editing Z (frames, bins) in place."""
    win = np.hanning(n + 1)[:-1]
    pad, view = _frames(x.astype(np.float64), n, hop)
    f = np.fft.rfftfreq(n, 1 / sr)
    t_all = frame_times(len(x), sr, n, hop)
    ola = np.zeros(len(pad))
    wsum = np.zeros(len(pad))
    for j0 in range(0, len(view), chunk):
        fr = view[j0:j0 + chunk]
        Z = np.fft.rfft(fr * win, axis=1)
        Z0 = Z.copy()
        fn(Z, t_all[j0:j0 + len(fr)], f)
        D = Z - Z0
        changed = np.flatnonzero(np.any(D != 0, axis=1))
        if len(changed):
            d = np.fft.irfft(D[changed], n=n, axis=1) * win
            for k, row in zip(changed, d):
                s = (j0 + k) * hop
                ola[s:s + n] += row
        for k in range(len(fr)):
            s = (j0 + k) * hop
            wsum[s:s + n] += win * win
    out = np.where(wsum > 1e-8, ola / np.maximum(wsum, 1e-8), 0.0)
    return out[n:n + len(x)]
