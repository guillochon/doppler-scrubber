"""doppler-scrubber command line."""
from __future__ import annotations

import argparse
import sys
import time

from . import scrub as S


def main(argv=None):
    ap = argparse.ArgumentParser(prog="doppler-scrubber",
                                 description="Remove aircraft Doppler tones from audio/video.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrub", help="write a copy of IN with Doppler tracks flattened")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--report", help="write tracks and stats as JSON (note: it describes the removed flyovers)")
    s.add_argument("--audio-only", action="store_true", help="drop video even if the output container allows it")
    s.add_argument("--passes", type=int, default=2); s.add_argument("--fmax", type=float, default=S.ScrubConfig.fmax)
    s.add_argument("--extend", type=float, default=90.0, help="seconds to clean beyond each visible track")
    s.add_argument("--thresh", type=float, default=6.0, help="peak threshold, dB above per-bin median")

    c = sub.add_parser("scan", help="list Doppler tracks without writing anything")
    c.add_argument("input"); c.add_argument("--fmax", type=float, default=S.ScrubConfig.fmax)
    c.add_argument("--thresh", type=float, default=6.0)

    a = ap.parse_args(argv)
    if a.cmd == "scrub":
        cfg = S.ScrubConfig(passes=a.passes, fmax=a.fmax, extend_s=a.extend, thresh_db=a.thresh, processes=True)
        t0 = time.time()
        def log(m):
            print(f"[{time.time() - t0:6.0f} s] {m}", file=sys.stderr, flush=True)
        r = S.scrub_file(a.input, a.output, cfg, report=a.report, keep_video=not a.audio_only, log=log)
        st = r["stats"]
        print(f"{st['tracks']} Doppler tracks flattened in {st['passes']} passes; "
              f"change {st['delta_rms_db_re_signal']:.1f} dB re signal; wrote {a.output}")
    else:
        cfg = S.ScrubConfig(fmax=a.fmax, thresh_db=a.thresh, processes=True)
        sr, ch = S._probe(a.input)
        low = S._decode(a.input, S.LOW_SR, ch).mean(axis=0)
        tracks = S.scan(low, S.LOW_SR, cfg)
        print(f"{'t0':>8} {'t1':>8} {'f0 Hz':>7} {'v m/s':>6} {'d m':>7} {'rms':>5} {'sweep':>6} {'dB':>5}")
        for t in tracks:
            print(f"{t.t0:8.1f} {t.t1:8.1f} {t.f0:7.1f} {t.v:6.0f} {t.d:7.0f} {t.rms:5.2f} {t.sweep:6.1f} {t.snr_db:5.1f}")
        if not tracks:
            print("no Doppler tracks found")


if __name__ == "__main__":
    main()
