#!/usr/bin/env python3
"""Spec-driven renderer.

  python3 engine/render.py scene.json preview 0 2.4 7 13.8     # full-size PNG stills -> <work>/preview/
  python3 engine/render.py scene.json full [--workers 2]        # all frames -> segments -> out/<name>.mp4 (+audio if present)

Layer order per frame:
  clean background plate -> bg effects (sky, water, flow, rotating rings) -> back particles -> objects (z order, after
  object effects: sway/bob) -> post effects (composite warps) -> additive light (glows, rays, embers, glints)
  -> front particles -> final look (bloom, vignette, grade, camera) -> frame lock
"""
import sys, os, subprocess, time, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import Scene
import effects as E


class Renderer:
    def __init__(self, path):
        self.sc = sc = Scene(path)
        self.fx = []
        for p in sc.spec.get('effects', []):
            if p.get('off'): continue
            if p['type'] not in E.REG:
                raise SystemExit('unknown effect type: %s (have: %s)' % (p['type'], ', '.join(sorted(E.REG))))
            self.fx.append(E.REG[p['type']](sc, p))
        has = lambda st: [e for e in self.fx if hasattr(e, st)]
        self.bg, self.back, self.obj, self.post = has('bg'), has('back'), has('obj'), has('post')
        self.light, self.front = has('light'), has('front')
        self.final = sorted(has('final'), key=lambda e: e.stage_prio)
        self.order = sorted(sc.objects, key=lambda n: sc.objects[n]['z'])
        if sc.frame is not None and not any(isinstance(e, E.FrameLock) for e in self.final):
            self.final.append(E.FrameLock(sc, {}))     # a frame mask exists -> always keep it still

    def render(self, t):
        sc = self.sc
        base = sc.bg.copy()
        for e in self.bg: base = e.bg(base, t)
        for e in self.back: e.back(base, t)
        layers = {n: sc.objects[n]['layer'] for n in self.order}
        for e in self.obj: e.obj(layers, t)
        for n in self.order:
            L = layers[n]; base = base * (1 - L[..., 3:4]) + L[..., :3]
        for e in self.post: base = e.post(base, t)
        if self.light:
            acc = np.zeros_like(base)
            for e in self.light: e.light(acc, t)
            base = base + acc
        for e in self.front: e.front(base, t)
        base = np.clip(base, 0, 255)
        for e in self.final: base = np.clip(e.final(base, t), 0, 255)
        return base.astype(np.uint8)


def render_chunk(args):
    path, i, f0, f1, outdir = args
    r = Renderer(path); sc = r.sc
    tmp = os.path.join(outdir, 'seg%d.tmp.mp4' % i); fin = os.path.join(outdir, 'seg%d.mp4' % i)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (sc.W, sc.H),
           '-r', str(sc.fps), '-i', '-', '-c:v', 'libx264', '-crf', '17', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
           '-g', str(sc.fps), '-an', tmp]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(f0, f1):
        pr.stdin.write(r.render(k / sc.fps).tobytes())
    pr.stdin.close(); rc = pr.wait()
    if rc == 0: os.replace(tmp, fin)      # a stopped render never leaves a short file under the final name
    return i, rc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('scene'); ap.add_argument('mode', choices=['preview', 'full'])
    ap.add_argument('times', nargs='*', type=float); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--segments', type=int, default=0)
    a = ap.parse_args()
    r = Renderer(a.scene); sc = r.sc
    if a.mode == 'preview':
        d = os.path.join(sc.work, 'preview'); os.makedirs(d, exist_ok=True)
        for t in (a.times or [0.0, sc.T * 0.25, sc.T * 0.5, sc.T * 0.9]):
            t0 = time.time(); fr = r.render(t); fn = os.path.join(d, 'f_%05.2f.png' % t)
            cv2.imwrite(fn, fr); print(fn, '%.2fs' % (time.time() - t0))
        return
    import multiprocessing as mp
    outdir = os.path.join(sc.work, 'seg'); os.makedirs(outdir, exist_ok=True)
    nseg = a.segments or max(a.workers * 2, 2); n = sc.nframes; step = -(-n // nseg)
    jobs = [(os.path.abspath(a.scene), i, i * step, min(n, (i + 1) * step), outdir) for i in range(nseg) if i * step < n]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(render_chunk, jobs, chunksize=1)
    if any(rc for _, rc in res): raise SystemExit('a segment failed: %s' % res)
    lst = os.path.join(outdir, 'list.txt')
    open(lst, 'w').write(''.join("file 'seg%d.mp4'\n" % i for i, _ in sorted(res)))
    outp = os.path.join(sc.dir, sc.spec.get('output', 'out/reel.mp4')); os.makedirs(os.path.dirname(outp), exist_ok=True)
    silent = outp + '.silent.tmp.mp4'
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', silent])
    wav = os.path.join(sc.work, 'audio.wav'); tmp = outp + '.tmp.mp4'
    if os.path.exists(wav):
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-i', wav, '-map', '0:v', '-map', '1:a',
                               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
                               '-shortest', '-movflags', '+faststart', tmp])
        os.remove(silent)
    else:
        os.replace(silent, tmp)
    os.replace(tmp, outp)
    print('done', outp, '%.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
