#!/usr/bin/env python3
"""Procedural sound bed (no samples, no licensing). Writes <work>/audio.wav, which render.py muxes automatically.

  python3 tools/audio.py scene.json

scene.json  "audio": {
   "mood": "devotional" | "nature" | "cinematic",
   "layers": ["pluck","drone","wind","water","rhythm","shimmer"],     # pick any; default depends on mood
   "root_hz": 130.81,                                                  # tonic for pluck / drone
   "cues": [ {"t": 2.25, "type": "bell"}, {"t": 5, "type": "hit"}, {"t": 9, "type": "riser"} ],
   "rhythm_from": 3.2,                                                 # damaru / pulse starts here (s)
   "gain": 0.8 }
Or  "audio": {"file": "my_music.mp3"}  to use your own track (the muxer expects work/audio.wav: it is converted for you).
Instagram note: for monetised / trending-audio Reels, export silent and add the platform track inside the app.
"""
import sys, os, json, subprocess
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile

SR = 44100


def main():
    path = sys.argv[1]; spec = json.load(open(path)); d = os.path.dirname(os.path.abspath(path))
    work = os.path.join(d, spec.get('work_dir', 'work')); os.makedirs(work, exist_ok=True)
    a = spec.get('audio')
    if not a: print('no "audio" in scene.json'); return
    wav = os.path.join(work, 'audio.wav')
    if 'file' in a:
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', os.path.join(d, a['file']), '-t', str(spec.get('duration', 14)),
                               '-ar', str(SR), '-ac', '2', wav]); return
    T = float(spec.get('duration', 14)); n = int(SR * T); t = np.arange(n) / SR; rng = np.random.RandomState(1)
    mood = a.get('mood', 'devotional'); root = a.get('root_hz', 130.81)
    defaults = {'devotional': ['pluck', 'drone', 'wind', 'shimmer'], 'nature': ['wind', 'water', 'drone'], 'cinematic': ['drone', 'rhythm', 'shimmer', 'wind']}
    layers = a.get('layers', defaults.get(mood, ['drone', 'wind']))
    L = np.zeros(n); R = np.zeros(n)

    def put(sig, t0, pan=0.5, gain=1.0):
        i = int(t0 * SR); m = min(len(sig), n - i)
        if m > 0: L[i:i + m] += sig[:m] * gain * (1 - pan) * 1.4; R[i:i + m] += sig[:m] * gain * pan * 1.4

    def bp(x, lo, hi, o=2): b_, a_ = butter(o, [lo / (SR / 2), hi / (SR / 2)], 'band'); return lfilter(b_, a_, x)
    def lp(x, fc, o=2): b_, a_ = butter(o, fc / (SR / 2), 'low'); return lfilter(b_, a_, x)

    def pluck(f, dur=5.5):                                    # tanpura / sitar-like: harmonics with slow 'jivari' shimmer
        tt = np.arange(int(SR * dur)) / SR; s = np.zeros_like(tt)
        for h in range(1, 18):
            s += (1.0 / h ** 0.9) * (1 + 0.6 * np.sin(TAU * (0.4 + 0.13 * h) * tt + h)) * np.sin(TAU * f * h * (1 + rng.uniform(-0.0012, 0.0012)) * tt) * np.exp(-tt * (0.55 + 0.18 * h))
        return s * np.minimum(tt / 0.01, 1) / np.abs(s).max()

    def bell(f, dur=6.0):
        tt = np.arange(int(SR * dur)) / SR; s = np.zeros_like(tt)
        for r_, am, dc in [(0.5, .6, .5), (1.0, 1, .6), (1.19, .7, .8), (1.71, .5, 1.1), (2.0, .6, 1.0), (2.74, .4, 1.6), (3.0, .3, 1.7), (3.76, .25, 2.1), (4.07, .2, 2.3)]:
            s += am * np.sin(TAU * f * r_ * tt + rng.uniform(0, 6)) * np.exp(-tt * dc)
        return s * np.minimum(tt / 0.003, 1) / np.abs(s).max()

    def hit(f0, dur=0.16):                                    # small drum / damaru
        tt = np.arange(int(SR * dur)) / SR; fr = f0 * (1 + 0.9 * np.exp(-tt * 35))
        return np.sin(TAU * np.cumsum(fr) / SR) * np.exp(-tt * 28) + bp(rng.randn(len(tt)), 1200, 4500) * np.exp(-tt * 120) * 0.35

    def boom(f=55, dur=2.5):
        tt = np.arange(int(SR * dur)) / SR
        return (np.sin(TAU * f * (1 + 2 * np.exp(-tt * 9)) * tt) * np.exp(-tt * 1.6)) * 1.0

    if 'pluck' in layers:
        seq = [root * 1.5, root * 2, root * 2, root]; k = 0; tp = 0.0
        while tp < T - 0.5:
            put(pluck(seq[k % 4]), tp, 0.35 + 0.3 * (k % 2), 0.26); tp += 1.55; k += 1
    if 'drone' in layers:
        pad = sum((1 / h) * np.sin(TAU * root / 2 * h * t + h) for h in range(1, 7)) * (0.5 + 0.5 * np.sin(TAU * t / 9.0 - 1.0)) ** 0.7
        pad = lp(pad, 900) * np.minimum(t / 2.0, 1); L += pad * 0.12; R += pad * 0.12
    if 'shimmer' in layers:
        sh = bp(rng.randn(n), 3500, 9000, 2) * (0.35 + 0.65 * np.exp(-((t - T * 0.2) / 0.9) ** 2)) * 0.03; L += sh; R += sh[::-1]
    if 'rhythm' in layers:
        ts = a.get('rhythm_from', 3.2); beat = 0.46; i = 0
        while ts < T - 1.2:
            for off, am, f0 in [(0, 1.0, 190), (0.5, .55, 230), (0.75, .5, 230), (1.0, .9, 190), (1.5, .5, 230)]:
                tp = ts + off * beat
                if tp < T - 0.8: put(hit(f0), tp, 0.5 + 0.12 * np.sin(i), 0.16 * min(1, (ts - ts * 0.9) / 3.0 + 0.35) * am)
                i += 1
            ts += 2 * beat; beat = max(beat * 0.985, 0.34)
    if 'wind' in layers:
        w = bp(rng.randn(n), 250, 1400, 2) * (0.5 + 0.5 * np.sin(TAU * t / (T / 2.5) + 0.8)) ** 1.5 * 0.07
        w2 = bp(rng.randn(n), 500, 2500, 2) * (0.5 + 0.5 * np.sin(TAU * t / (T / 2.0) + 2.2)) ** 2 * 0.04
        L += w + w2 * 0.6; R += w * 0.8 + w2
    if 'water' in layers:
        wt = (lp(rng.randn(n), 3500) * 0.5 + bp(rng.randn(n), 4000, 8000) * 0.35) * 0.05 * (0.75 + 0.25 * np.sin(TAU * t * 1.7) * np.sin(TAU * t * 0.9 + 1))
        L += wt * 1.1; R += wt * 0.8
    for c in a.get('cues', []):                                # one-shot accents: line them up with a visual event (flash, reveal)
        ty = c['type']; t0 = c['t']
        if ty == 'bell': put(bell(c.get('hz', 392.0)), t0, 0.5, c.get('gain', 0.34))
        elif ty == 'hit': put(hit(c.get('hz', 190)), t0, 0.5, c.get('gain', 0.4))
        elif ty == 'boom': put(boom(c.get('hz', 55)), t0, 0.5, c.get('gain', 0.5))
        elif ty == 'chime': put(bell(c.get('hz', 880), 3.5), t0, 0.55, c.get('gain', 0.2))
        elif ty == 'riser':
            dur = c.get('dur', 2.5); i0 = int(t0 * SR); m = min(int(dur * SR), n - i0); u = np.linspace(0, 1, m)
            r_ = bp(rng.randn(m), 600, 6000) * u ** 2 * 0.12; L[i0:i0 + m] += r_; R[i0:i0 + m] += r_
        elif ty == 'swell':
            dur = c.get('dur', 2.0); i0 = int(t0 * SR); m = min(int(dur * SR), n - i0); u = np.sin(np.linspace(0, np.pi, m))
            r_ = bp(rng.randn(m), 3500, 9000) * u * 0.06; L[i0:i0 + m] += r_; R[i0:i0 + m] += r_[::-1]
    fade = np.minimum(t / 0.4, 1) * np.minimum((T - t) / 1.2, 1)
    L *= fade; R *= fade; pk = max(np.abs(L).max(), np.abs(R).max()) + 1e-9; sc = a.get('gain', 0.8) / pk
    wavfile.write(wav, SR, (np.stack([L * sc, R * sc], 1) * 32767).astype(np.int16)); print('audio ->', wav, 'layers', layers)


TAU = 2 * np.pi
if __name__ == '__main__':
    main()
