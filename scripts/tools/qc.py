#!/usr/bin/env python3
"""Automatic checks on a finished reel + contact sheet.

  python3 tools/qc.py scene.json [video.mp4]

Checks: size / fps / duration / audio stream, frame-lock really still (mean |diff| to the original inside the frame mask),
loop seam (first vs last frame), per-frame jump detector (segment joins), audio loudness. Writes work/qc_sheet.png
(original | t = 10 %, 40 %, 70 %, 95 %) so you can see the change at a glance.
"""
import sys, os, json, subprocess
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'engine'))
from core import Scene


def main():
    sc = Scene(sys.argv[1]); vid = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sc.dir, sc.spec.get('output', 'out/reel.mp4'))
    pr = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', vid]))
    v = [s for s in pr['streams'] if s['codec_type'] == 'video'][0]; au = [s for s in pr['streams'] if s['codec_type'] == 'audio']
    print('video %sx%s %s fps %s frames %s  audio: %s' % (v['width'], v['height'], v['r_frame_rate'], v.get('nb_frames'), v.get('duration'), 'yes' if au else 'NO'))
    cap = cv2.VideoCapture(vid); frames = []; prev = None; jumps = []; fr_diff = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f if len(frames) % 15 == 0 else None)
        if prev is not None: jumps.append(float(np.abs(f.astype(np.int16) - prev.astype(np.int16)).mean()))
        if sc.frame is not None and len(frames) % 30 == 1:
            m = sc.frame > 0.9; fr_diff.append(float(np.abs(f.astype(np.float32) - sc.upf)[m].mean()))
        prev = f; last = f
    first = next(x for x in frames if x is not None)
    print('frames read', len(frames))
    if jumps:
        j = np.array(jumps); print('frame-to-frame change: mean %.2f  max %.2f at frame %d (a joint/seam shows as a spike > 4x mean)' % (j.mean(), j.max(), j.argmax()))
    if fr_diff: print('frame-lock error (mean |video-original| inside frame mask): %.2f  (a few levels is codec noise on fine ornaments; >10 means the frame is moving)' % np.mean(fr_diff))
    print('loop seam: |last-first| = %.2f  (only meaningful if you designed a loop: camera.loop=true, rotate_ring ease_tau=0 and whole revolutions; a push-in reel is one-shot, so a large value is expected)' % np.abs(last.astype(np.int16) - first.astype(np.int16)).mean())
    if au:
        o = subprocess.run(['ffmpeg', '-i', vid, '-af', 'volumedetect', '-vn', '-f', 'null', '-'], capture_output=True, text=True).stderr
        print([l.strip() for l in o.splitlines() if 'mean_volume' in l or 'max_volume' in l])
    pick = [frames[int(len(frames) * q) // 15 * 15] for q in (0.1, 0.4, 0.7, 0.95)]
    tiles = [sc.up] + [p for p in pick if p is not None]
    sheet = np.hstack([cv2.resize(x, (360, 640)) for x in tiles]); cv2.imwrite(os.path.join(sc.work, 'qc_sheet.png'), sheet); print('sheet ->', sc.work + '/qc_sheet.png')


if __name__ == '__main__':
    main()
