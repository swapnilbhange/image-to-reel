#!/usr/bin/env python3
"""Find the exact centre and radius of a round thing (halo, mandala, wheel, clock) so it rotates without wobble.

  python3 tools/fit_circle.py image.png cx0 cy0 rmin rmax [deg0 deg1] [--mask occluder_mask.png]

Rough centre (cx0, cy0) and radius search band [rmin, rmax] come from the grid sheet. For every angle in [deg0, deg1]
(image angles, y down) the strongest outward brightness drop in the band is taken as a rim point (angles whose ray
touches the optional mask are skipped); a least-squares circle is fitted with iterative outlier rejection.
Prints  cx cy R  (SOURCE pixels) and writes circle_overlay.png.  Check the overlay: the green circle must sit on the rim.
"""
import sys, cv2, numpy as np

args = [a for a in sys.argv[1:] if not a.startswith('--')]
src = cv2.imread(args[0]); H, W = src.shape[:2]
cx0, cy0, rmin, rmax = map(float, args[1:5]); d0, d1 = (float(args[5]), float(args[6])) if len(args) > 6 else (0.0, 360.0)
g = cv2.GaussianBlur(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
occ = np.zeros((H, W), bool)
if '--mask' in sys.argv:
    occ = cv2.dilate((cv2.imread(sys.argv[sys.argv.index('--mask') + 1], 0) > 127).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
pts = []
for deg in np.arange(d0, d1, 1.0):
    t = np.deg2rad(deg); rs = np.arange(rmin, rmax, 0.5)
    xs, ys = cx0 + rs * np.cos(t), cy0 + rs * np.sin(t); xi, yi = xs.round().astype(int), ys.round().astype(int)
    if (xi < 0).any() or (xi >= W).any() or (yi < 0).any() or (yi >= H).any() or occ[yi, xi].any(): continue
    k = np.diff(g[yi, xi]).argmin(); pts.append((xs[k], ys[k]))
pts = np.array(pts); print(len(pts), 'rim points')


def fit(p):
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]; b = p[:, 0] ** 2 + p[:, 1] ** 2
    c = np.linalg.lstsq(A, b, rcond=None)[0]; return c[0], c[1], np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)


p = pts
for _ in range(6):
    cx, cy, R = fit(p); res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - R)
    p = pts[res < max(1.5, np.percentile(res, 70))]
cx, cy, R = fit(p); print('centre %.2f %.2f   radius %.2f' % (cx, cy, R))
ov = src.copy(); cv2.circle(ov, (round(cx), round(cy)), round(R), (0, 255, 0), 1); cv2.circle(ov, (round(cx), round(cy)), 3, (0, 255, 0), -1)
cv2.imwrite('circle_overlay.png', ov)
