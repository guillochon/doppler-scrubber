"""Compare an original recording with its scrubbed copy.

    uv run python scripts/quality.py original.webm scrubbed.flac

Prints how much changed overall, where in frequency the change sits, and
what fraction of time-frequency cells moved by more than 1 dB.
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.signal as ss

from doppler_scrubber.scrub import _decode, _probe


def main(a: str, b: str):
    sr, ch = _probe(a)
    x = _decode(a, sr, ch)
    y = _decode(b, sr, ch)
    n = min(x.shape[1], y.shape[1])
    x, y = x[:, :n].astype(np.float64), y[:, :n].astype(np.float64)
    d = y - x
    print(f"{n / sr / 60:.1f} min, {sr} Hz, {ch} ch")
    print(f"change re signal, overall:     {10 * np.log10(np.mean(d ** 2) / np.mean(x ** 2)):7.1f} dB")
    f, Px = ss.welch(x, sr, nperseg=1 << 15, axis=1)
    _, Pd = ss.welch(d, sr, nperseg=1 << 15, axis=1)
    Px, Pd = Px.mean(0), Pd.mean(0)
    for lo, hi in [(0, 30), (30, 450), (450, 1000), (1000, 2000), (2000, sr / 2)]:
        m = (f >= lo) & (f < hi)
        print(f"change re signal, {lo:>5.0f}-{hi:<5.0f} Hz: {10 * np.log10(Pd[m].sum() / Px[m].sum() + 1e-30):7.1f} dB")
    # fraction of STFT cells (0.17 s x 5.9 Hz at 48 kHz) that moved by more than 1 dB
    nper = int(2 ** np.round(np.log2(sr * 0.17)))
    _, _, X = ss.stft(x[0], sr, nperseg=nper)
    _, _, Y = ss.stft(y[0], sr, nperseg=nper)
    dl = np.abs(10 * np.log10((np.abs(Y) ** 2 + 1e-12) / (np.abs(X) ** 2 + 1e-12)))
    loud = np.abs(X) ** 2 > 1e-10
    print(f"STFT cells changed > 1 dB:     {100 * np.mean(dl[loud] > 1.0):7.3f} %  (of cells above -100 dB)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
