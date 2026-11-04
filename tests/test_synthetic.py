import numpy as np
import scipy.ndimage as nd
import scipy.signal as ss

from doppler_scrubber import ScrubConfig, scrub_array, stft
from doppler_scrubber.tracks import ideal_curve

SR = 4000


def _tone(t, f0, v, d, tc, amp):
    finst = np.nan_to_num(ideal_curve(t, f0, v, d, tc), nan=f0)
    return amp * np.sin(2 * np.pi * np.cumsum(finst) / SR)


def _along(x, f0, v, d, tc, offset=0.0, t_lo=30, t_hi=270):
    t, f, L = stft.power_db(x, SR, 8192, 1000, 40, 450)
    W = L - np.median(L, axis=1, keepdims=True)
    m = (t > t_lo) & (t < t_hi)
    fc = ideal_curve(t[m], f0, v, d, tc) + offset
    j = np.argmin(np.abs(f[:, None] - fc[None, :]), axis=0)
    return float(W[j, np.flatnonzero(m)].mean())


def test_removes_doppler_tone_and_keeps_rest():
    # reference: whitened level along the same curve in noise alone is about -0.5 dB
    rng = np.random.default_rng(1)
    t = np.arange(0, 300, 1 / SR)
    noise = 0.3 * rng.standard_normal(len(t))
    hum = 0.05 * np.sin(2 * np.pi * 120.0 * t)          # stationary tone: must survive
    plane = _tone(t, 140.0, 70.0, 1500.0, 150.0, 0.03) + _tone(t, 280.0, 70.0, 1500.0, 150.0, 0.015)
    x = noise + hum + plane

    before = _along(x, 140.0, 70.0, 1500.0, 150.0)
    delta, tracks, stats = scrub_array(x[None, :], SR, ScrubConfig())
    y = x + delta[0]
    after = _along(y, 140.0, 70.0, 1500.0, 150.0)
    after_h2 = _along(y, 280.0, 70.0, 1500.0, 150.0)

    assert len(tracks) >= 1
    assert before > 8.0
    assert after < 2.0 and after_h2 < 2.0
    # no dark notch either: the corridor sits at the noise floor
    assert after > -2.5 and after_h2 > -2.5
    # hum untouched
    f, P = np.fft.rfftfreq(len(y), 1 / SR), np.abs(np.fft.rfft(y)) ** 2
    f, P0 = np.fft.rfftfreq(len(x), 1 / SR), np.abs(np.fft.rfft(x)) ** 2
    k = np.argmin(np.abs(f - 120.0))
    assert abs(10 * np.log10(P[k] / P0[k])) < 0.5
    # the output looks like the recording without the plane: its long-term spectrum, in 5 Hz
    # bands, matches noise + hum alone (the band replacement swaps noise for like noise) ...
    clean = noise + hum
    fb = np.arange(30, 1000, 5.0)
    fq = np.fft.rfftfreq(len(y), 1 / SR)
    band = np.digitize(fq, fb)
    sel = (band > 0) & (band < len(fb))
    by = np.bincount(band[sel], (np.abs(np.fft.rfft(y)) ** 2)[sel])[1:]
    bc = np.bincount(band[sel], (np.abs(np.fft.rfft(clean)) ** 2)[sel])[1:]
    assert np.max(np.abs(10 * np.log10(by / bc))) < 1.0
    # ... and almost none of the change lands where the plane has no energy: 600-1000 Hz
    # holds only empty harmonic lines, which are left alone
    d = delta[0]
    fd, Pd = np.fft.rfftfreq(len(d), 1 / SR), np.abs(np.fft.rfft(d)) ** 2
    far = (fd > 600) & (fd < 1000)
    near = (fd > 100) & (fd < 200)
    assert Pd[far].mean() < 1e-2 * Pd[near].mean()


def test_removes_tone_too_faint_for_frame_peaks():
    # 0.02 is the faintest this pass can be and still be detected by dopplertrack (at 0.015
    # it finds nothing; at 0.02 it fits the pass exactly, score 6.2), so it is the faintest
    # tone an attacker can use: the scrubber must find and remove it
    rng = np.random.default_rng(3)
    t = np.arange(0, 400, 1 / SR)
    x = 0.3 * rng.standard_normal(len(t)) + _tone(t, 130.0, 80.0, 3000.0, 200.0, 0.02)
    before = _along(x, 130.0, 80.0, 3000.0, 200.0, t_lo=120, t_hi=300)
    delta, tracks, _ = scrub_array(x[None, :], SR, ScrubConfig())
    after = _along(x + delta[0], 130.0, 80.0, 3000.0, 200.0, t_lo=120, t_hi=300)
    assert before > 2.0
    assert len(tracks) >= 1
    assert after < 0.5 * before


def test_noise_only_is_left_alone():
    x = 0.3 * np.random.default_rng(2).standard_normal(SR * 200)
    delta, tracks, _ = scrub_array(x[None, :], SR, ScrubConfig())
    assert tracks == []
    assert not np.any(delta)


def test_no_whistle_above_fmax():
    # a pass whose harmonics sweep across fmax, over a background that is loud at low
    # frequencies and quiet above (as outdoors): nothing above fmax may change, and no band
    # may come out louder than it was (a 1-2 Hz band of extra noise is heard as a whistle)
    rng = np.random.default_rng(4)
    t = np.arange(0, 300, 1 / SR)
    low = ss.sosfilt(ss.butter(4, 300, fs=SR, output="sos"), rng.standard_normal(len(t)))
    x = 0.5 * low + 0.003 * rng.standard_normal(len(t))
    x += sum(_tone(t, 150.0 * h, 90.0, 800.0, 150.0, 0.03 / h) for h in range(1, 9))
    cfg = ScrubConfig(fmax=1000.0)
    delta, tracks, _ = scrub_array(x[None, :], SR, cfg)
    assert len(tracks) >= 1
    d = delta[0]
    fd = np.fft.rfftfreq(len(d), 1 / SR)
    P, Pd = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(d)) ** 2
    above = fd > cfg.fmax + 20
    assert Pd[above].sum() < 1e-4 * P[above].sum()
    # 30 s-smoothed power per ~0.5 Hz bin, output vs input: a whistle is a band that stays
    # louder (the old code: +16 dB above fmax, 99.9th percentile +4.9 dB); a noise-for-noise
    # swap only fluctuates (over 5 s it passes +6 dB in a few % of the cells it replaces)
    _, f, L0 = stft.power_db(x, SR, 8192, 1000, 30, 2000)
    _, _, L1 = stft.power_db(x + d, SR, 8192, 1000, 30, 2000)
    p0 = nd.uniform_filter1d(10 ** (L0 / 10), 120, axis=1)
    p1 = nd.uniform_filter1d(10 ** (L1 / 10), 120, axis=1)
    r = 10 * np.log10(p1 / p0)
    assert r[f > cfg.fmax + 20].max() < 1.0
    assert np.percentile(r, 99.9) < 3.5
