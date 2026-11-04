# doppler-scrubber

Remove aircraft Doppler tones from audio and video, so a recording cannot be
geolocated from them, without otherwise changing how it sounds.

Outdoor recordings carry faint tones from passing propeller aircraft (blade
pass and its harmonics, roughly 40–1000 Hz). The way each tone falls in
pitch as the aircraft passes depends on where the microphone was, and
matching that curve against public ADS-B flight history can place a
recording to within tens of metres
([dopplertrack](https://github.com/mjshiggins/dopplertrack) is an open
implementation). This package finds those tones and removes them.

## Usage

```bash
uv sync
uv run doppler-scrubber scan recording.mp4                 # list flyover tones, change nothing
uv run doppler-scrubber scrub recording.mp4 clean.mp4      # video stream copied, audio re-encoded once
uv run doppler-scrubber scrub recording.wav clean.flac --report report.json
```

The report lists the removed tracks with their fitted speed, distance and
time of closest approach, which is itself location-revealing: keep it private.

## How it works

1. **Find, two ways.** A 2 s STFT of a 4 kHz mono copy is whitened per
   frequency bin.
   - *Ridge tracks:* narrowband peaks are linked frame to frame, and a track
     is kept only if a straight-pass Doppler model (rest frequency, speed,
     closest-approach distance and time, with propagation delay, in closed
     form) fits it to a fraction of a hertz. Speech and music harmonics, hum,
     birdsong and noise chains fail that test. Each kept track is followed
     outward along its own model and refit, which joins fragments of one pass
     and follows faint tails.
   - *Matched-filter tracks:* the attacker's view. Every curve on a
     (speed, distance, CPA time, frequency) grid is scored by its ridge
     contrast integrated over a 4-minute window, and the best are polished
     with Nelder-Mead. This finds tones far too faint to show frame by frame,
     and anything scoring 2 dB or more is removed.
2. **Remove.** Each line (the track times every ratio m/h, h ≤ 2, up to
   `fmax`) is heterodyned to 0 Hz with its tracked phase and low-passed to get
   the tone's complex amplitude, with small frequency errors corrected from
   the phase drift. Over the track's span the band is *replaced*: the
   amplitude is subtracted in full and independent noise at the level of side
   bands 4–6 Hz away is put back, so the band holds noise of the right level
   and no tone at all, whatever its SNR. The band is about 1 Hz wide for a
   measured ridge and about 2.5 Hz for a matched-filter model. Harmonic lines
   other than the detected one are replaced only if, over the whole span,
   they hold clearly more power than their side bands.
3. **Repeat** detection on the cleaned audio (2 passes) to catch tones that
   were hidden under stronger ones.
4. **Protect steady lines.** Mains hum and machinery (bins loud at their 20th
   percentile) are taken out before scrubbing and are never edited, so a
   flyover crossing them leaves them as they were.
5. **Write.** The change is computed at 4 kHz, upsampled and added to the
   original full-rate samples, so content above ~1.9 kHz is untouched;
   nothing more than ~1.3 Hz from a removed line changes.

The matched-filter threshold is set per recording from a null: the same
search on the time-reversed spectrogram, where a real pass would sweep *up*,
which no aircraft does. It is kept within 2.0–2.5 dB.

## Results

Tested against dopplertrack on its own two validation recordings, attacking the
scrubbed audio with the event fits from the *original* audio (more than a real
attacker would have).

| | Oyster Bay, NY (1 h) | SW Florida eagle cam (23 min clip) |
|---|---|---|
| dopplertrack `detect`, before | flyovers ranked 1 and 2 (15.7, 10.0) | flyovers ranked 1 and 3 (4.9, 3.7) |
| dopplertrack `detect`, after | gone; best curve anywhere 2.9 | gone; best curve anywhere 1.7 |
| known-point score of the true flight pair, before | 28.0 (the only pair) | 9.5, best wrong pair 6.7 |
| known-point score of the true flight pair, after | 5.6 | 5.2, best wrong pair 6.6 |
| blind 200 m search, before (dopplertrack README) | true pair 1st of 328, 43 m off after refining | true pair 1st of 607 |
| blind 200 m search, after | true pair 59th of 328, top guess 31 km away | not run |
| tracks removed / time | 330 / 12.4 min | 85 / 3.2 min |
| change above 2 kHz | −40 dB re signal | −62 dB re signal |
| STFT cells changed by > 1 dB | 25 % | 5.8 % |

The cost depends on how busy the recording is. Oyster Bay, under New York
approach paths, is full of Doppler-shaped tones, and the scrubber replaces a
quarter of the 30–1000 Hz time-frequency cells there (with noise of the same
level); a quieter recording loses far less. Every change is narrowband noise
traded for like noise, but listen to the result before relying on it.

## Examples

`examples/` holds 60 s before/after clips (FLAC, 48 kHz stereo) from the two
test recordings:

| clip | source | span |
|---|---|---|
| `oyster_A_original.flac` / `oyster_B_scrubbed.flac` | Oyster Bay, NY ([4CxVxuMVkMc](https://www.youtube.com/watch?v=4CxVxuMVkMc)) | 35:25–36:25 |
| `eaglecam_A_original.flac` / `eaglecam_B_scrubbed.flac` | SW Florida Eagle Cam ([zs2o5Rhlp5w](https://www.youtube.com/watch?v=zs2o5Rhlp5w)) | 5:37:00–5:38:00, around a flyover's closest approach |

Both locations are already public: the first is dopplertrack's own reference
recording and the second is the nest's published position.

## Limitations

- Only aircraft-like Doppler curves (straight passes, 20–250 m/s) are looked
  for. Turning aircraft are tracked by the ridge follower but fit worse.
- The threshold is tuned against dopplertrack's detector. A better detector
  could find fainter curves than this one removes.
- Speed: roughly 12 minutes per hour of stereo audio on a busy recording,
  3 minutes per 20 minutes on a quiet one.

## Development

```bash
uv run pytest
```
