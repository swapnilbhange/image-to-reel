#!/usr/bin/env python3
"""Step 1 of the flow: look at the image like a compositor.

  python3 tools/inspect_image.py image.png [outdir=inspect]

Writes into outdir:
  overview.png      whole image with a coordinate grid every 100 px (source pixels) - read positions from here
  tile_*.png        2x zoomed horizontal bands with a 50 px grid - for exact points (SAM prompts, polygons, glow centres)
  regions.png       auto-guessed regions painted over the image (sky / water / white-water / light sources / frame / circles / faces)
  inspect.json      the same guesses as numbers, to paste into scene.json (they are SUGGESTIONS - confirm by eye)

What it looks for, and why:
  frame        a decorative border that should stay still (gold / dark band at the edge)         -> frame_lock
  circles      big round structures: halos, mandalas, wheels, moon, sun                           -> rotate_ring / rays / glow
  sky          smooth, bluish/grey/orange top region with low texture                             -> sky_drift
  water        lower region, blue/teal, wide and horizontal                                       -> water_ripple
  white water  bright, unsaturated, tall thin shapes                                              -> flow (waterfall / mist)
  light blobs  small very bright spots (flames, jewels, lamps, stars, eyes)                       -> glow / flicker / rays
  faces        frontal faces (OpenCV Haar)                                                         -> keep still, glow eyes, no warps
  saliency     what the eye is drawn to                                                           -> what must NOT be warped
"""
import sys, os, json
import numpy as np, cv2


def grid(img, step, scale=1.0, x0=0, y0=0, font=0.45):
    o = img.copy(); h, w = o.shape[:2]
    for x in range((x0 // step + 1) * step, x0 + int(w / scale), step):
        X = int((x - x0) * scale); cv2.line(o, (X, 0), (X, h), (255, 255, 255), 1)
        cv2.putText(o, str(x), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range((y0 // step + 1) * step, y0 + int(h / scale), step):
        Y = int((y - y0) * scale); cv2.line(o, (0, Y), (w, Y), (255, 255, 255), 1)
        cv2.putText(o, str(y), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 0), 1, cv2.LINE_AA)
    return o


def biggest(mask, minarea):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = [(st[i, 4], st[i, :4].tolist(), cen[i].tolist()) for i in range(1, n) if st[i, 4] >= minarea]
    return sorted(comps, reverse=True)


def main():
    path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 'inspect'; os.makedirs(out, exist_ok=True)
    src = cv2.imread(path); H, W = src.shape[:2]; res = {'size': [W, H]}
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV); h_, s_, v_ = [hsv[..., i].astype(int) for i in range(3)]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # ---- grids
    sc = min(1.0, 1600 / H)
    cv2.imwrite(os.path.join(out, 'overview.png'), grid(cv2.resize(src, None, fx=sc, fy=sc), 100, sc))
    band = 360
    for i, y0 in enumerate(range(0, H, band)):
        crop = src[y0:y0 + band + 40]; z = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out, 'tile_%02d_y%d.png' % (i, y0)), grid(z, 50, 2.0, 0, y0))
    ov = src.copy()
    # ---- frame (border band that differs from the interior: gold-ish or very dark/saturated)
    edge = 0.03; bw = max(8, int(W * edge))
    ring = np.zeros((H, W), bool); ring[:bw] = ring[-bw:] = 1; ring[:, :bw] = ring[:, -bw:] = 1
    gold = (h_ >= 12) & (h_ <= 32) & (s_ >= 100) & (v_ >= 105)
    frac = float(gold[ring].mean()); res['frame'] = {'gold_fraction_on_border': round(frac, 2), 'likely_frame': frac > 0.25}
    if frac > 0.25:                       # measure thickness from each side
        def thick(line):
            k = 0
            while k < len(line) // 3 and line[k]: k += 1
            return k
        gl = cv2.morphologyEx(gold.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        cols = gl[H // 3:2 * H // 3]; rows = gl[:, W // 3:2 * W // 3]
        res['frame']['border_px_l_t_r_b_suggested'] = [int(np.median([thick(r) for r in cols])) + 3, int(np.median([thick(c) for c in rows.T])) + 3,
                                                       int(np.median([thick(r[::-1]) for r in cols])) + 3, int(np.median([thick(c[::-1]) for c in rows.T])) + 3]
        ov[gold & ring] = (0, 215, 255)
    # ---- circles
    g = cv2.GaussianBlur(gray, (0, 0), 3)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.5, minDist=min(H, W) * 0.15, param1=120, param2=60,
                         minRadius=int(min(W, H) * 0.12), maxRadius=int(max(W, H) * 0.45))
    res['circles'] = [] if c is None else [[round(float(x), 1), round(float(y), 1), round(float(r), 1)] for x, y, r in c[0][:6]]
    for x, y, r in res['circles']: cv2.circle(ov, (int(x), int(y)), int(r), (0, 255, 0), 2)
    res['circles_note'] = 'rough only - refine with tools/fit_circle.py'
    # ---- sky: top 45 %, low local texture, not too dark
    lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 6)
    sky = (lap < np.percentile(lap, 35)) & (v_ > 60); sky[int(H * 0.45):] = 0
    sk = biggest(cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)), W * H * 0.01)
    res['sky_candidates'] = [{'area_px': int(a), 'bbox_xywh': b} for a, b, _ in sk[:3]]
    for a, (x, y, w, h), _ in sk[:3]: cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 160, 0), 2)
    # ---- water: lower 40 %, blue/teal, wide
    wat = ((h_ >= 85) & (h_ <= 130) & (s_ > 40) & (v_ > 40)); wat[:int(H * 0.55)] = 0
    wc = biggest(cv2.morphologyEx(wat.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), W * H * 0.01)
    res['water_candidates'] = [{'area_px': int(a), 'bbox_xywh': b} for a, b, _ in wc[:3]]
    for a, (x, y, w, h), _ in wc[:3]: cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # ---- white water / mist: bright, unsaturated, tall
    ww = (v_ > 190) & (s_ < 60); ww = cv2.morphologyEx(ww.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    wl = [x for x in biggest(ww, 600) if x[1][3] > 1.4 * x[1][2]]
    res['white_water_candidates'] = [{'area_px': int(a), 'bbox_xywh': b} for a, b, _ in wl[:3]]
    for a, (x, y, w, h), _ in wl[:3]: cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # ---- light sources: small very bright & saturated-or-white blobs
    bl = (v_ > 235) & (s_ > 25); bl = cv2.morphologyEx(bl.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lb = [x for x in biggest(bl, 25) if x[0] < W * H * 0.01][:14]
    res['light_blobs'] = [{'xy': [round(c[0]), round(c[1])], 'area_px': int(a)} for a, b, c in lb]
    for a, b, c in lb: cv2.circle(ov, (int(c[0]), int(c[1])), 9, (0, 0, 255), 2)
    # ---- faces
    try:
        fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        fs = fc.detectMultiScale(gray, 1.1, 5, minSize=(int(W * 0.04),) * 2)
        res['faces_xywh'] = [list(map(int, f)) for f in fs]
        for x, y, w, h in fs: cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 0, 255), 2)
    except Exception as e:
        res['faces_xywh'] = []
    # ---- saliency
    try:
        ok, sal = cv2.saliency.StaticSaliencySpectralResidual_create().computeSaliency(src)
        sal = cv2.GaussianBlur((sal * 255).astype(np.uint8), (0, 0), 8); cv2.imwrite(os.path.join(out, 'saliency.png'), sal)
        ys, xs = np.where(sal > np.percentile(sal, 92)); res['saliency_centroid'] = [int(xs.mean()), int(ys.mean())]
    except Exception:
        pass
    # ---- mood hints
    res['palette_bgr_dominant'] = [int(x) for x in cv2.kmeans(cv2.resize(src, (64, 64)).reshape(-1, 3).astype(np.float32), 3, None,
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1), 3, cv2.KMEANS_PP_CENTERS)[2].flatten()]
    res['mean_brightness'] = round(float(v_.mean()), 1); res['mean_saturation'] = round(float(s_.mean()), 1)
    cv2.imwrite(os.path.join(out, 'regions.png'), grid(cv2.resize(ov, None, fx=sc, fy=sc), 100, sc))
    json.dump(res, open(os.path.join(out, 'inspect.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'palette_bgr_dominant'}, indent=1)[:2500])
    print('\nNow LOOK at overview.png, tile_*.png, regions.png and write the object inventory (docs/FLOW.md step 2).')


if __name__ == '__main__':
    main()
