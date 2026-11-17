#!/usr/bin/env python3
"""Build the 'plate' for every rotate_ring effect: the ring pattern with the parts hidden behind objects re-invented.

  python3 tools/polar_fill.py scene.json

For a rotate_ring effect with   "occluders": ["figure", ...]   (object names)   and optional
  "occluder_circles": [[x,y,r],..]   extra things that sit in front (src px)
  "bad_deg": [[a,b],..]              degrees (clockwise from 3 o'clock) that must NOT be used as donor (unique art: a face, a logo)
  "seam": 50                         overlap length in polar columns (bigger = smoother, slower)
the ring is unrolled with cv2.warpPolar (angle -> x, radius -> y), the columns touched by an occluder are filled
by quilting along the angle axis from fully visible columns of the same ring (normal + mirrored donors, used-donor penalty,
min-error seam), and rolled back. Output: <work>/plates/<id>.png  (full output size; only the ring area matters).
No occluder in the ring -> plain copy of the picture.
"""
import sys, os
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'engine'))
from core import Scene

TH = 3600


def seam_path(E):
    L, h = E.shape; cost = E.copy()
    for r in range(1, h):
        pc = cost[:, r - 1]
        cost[:, r] = E[:, r] + np.minimum(np.minimum(pc, np.r_[np.inf, pc[:-1]]), np.r_[pc[1:], np.inf])
    c = int(cost[:, -1].argmin()); path = np.zeros(h, int); path[-1] = c
    for r in range(h - 1, 0, -1):
        c = min([x for x in (c - 1, c, c + 1) if 0 <= x < L], key=lambda x: cost[x, r - 1]); path[r - 1] = c
    return path


def fill_band(P, occ, bad, pad=14, L=50, mmax=520):
    TH_, h, _ = P.shape
    col = occ.any(axis=1)
    colD = np.convolve(np.r_[col[-pad:], col, col[:pad]].astype(int), np.ones(2 * pad + 1, int), 'valid') > 0
    if not colD.any(): return P, colD, []
    if colD.all(): raise RuntimeError('ring fully occluded')
    f0 = int(np.where(~colD)[0][0]); cc = np.roll(colD, -f0); best, bs, i = 0, 0, 0
    while i < TH_:                                  # roll so a clear column sits at the array start (centre of the largest clear run)
        if not cc[i]:
            j = i
            while j + 1 < TH_ and not cc[j + 1]: j += 1
            if j - i + 1 > best: best, bs = j - i + 1, i
            i = j + 1
        else: i += 1
    c0 = (f0 + bs + best // 2) % TH_
    Pr = np.roll(P, -c0, axis=0); cD = np.roll(colD, -c0); badr = np.roll(bad, -c0)
    runs, i = [], 0
    while i < TH_:
        if cD[i]:
            j = i
            while j + 1 < TH_ and cD[j + 1]: j += 1
            runs.append([i, j]); i = j + 1
        else: i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 < 2 * L + 10: merged[-1][1] = r[1]
        else: merged.append(r)
    for a, b in merged: cD[a:b + 1] = True
    elig = (~cD) & (~badr); arrs = []
    for o in (0, 1):
        A = Pr if o == 0 else Pr[::-1]; E_ = elig if o == 0 else elig[::-1]
        arrs.append((np.concatenate([A, A, A], axis=0), np.r_[0, np.cumsum(np.r_[E_, E_, E_])]))
    used = np.zeros(TH_, bool); cur = Pr.copy(); log = []

    def idx(o, s, x0, x1):
        ii = np.arange(x0 + s, x1 + s) % TH_
        return ii if o == 0 else TH_ - 1 - ii
    for a, b in merged:
        p = a; prev = None
        while p <= b:
            m = min(mmax, b - p + 1); last = (p + m - 1 == b); ext = m + (L if last else 0)
            tgtL = cur[p - L:p]; tgtR = cur[b + 1:b + 1 + L] if last else None; cands = []
            for o in (0, 1):
                Pd, dcs = arrs[o]
                for s in range(TH_):
                    d0 = (p + s - L) % TH_
                    if dcs[d0 + L + ext] - dcs[d0] != L + ext: continue
                    c = ((tgtL - Pd[TH_ + p + s - L:TH_ + p + s]) ** 2).mean()
                    if last: c += ((tgtR - Pd[TH_ + b + 1 + s:TH_ + b + 1 + s + L]) ** 2).mean()
                    c *= (1 + 3 * used[idx(o, s, p, p + m)].mean())        # avoid repeating the same donor art
                    if prev == (o, s) or (prev and prev[0] == o and abs(((s - prev[1] + TH_ // 2) % TH_) - TH_ // 2) < 8): c *= 0.5
                    cands.append((c, o, s))
            if not cands: raise RuntimeError('no donor stretch long enough: reduce "seam" or the bad_deg ranges')
            cands.sort(); c_, o, s = cands[0]; prev = (o, s); Pd = arrs[o][0]; used[idx(o, s, p - L, p + ext)] = True
            log.append((int(a), int(b), int(p), int(m), int(o), int(s), round(float(c_), 1)))
            D = lambda x0, x1: Pd[TH_ + x0 + s:TH_ + x1 + s]
            path = seam_path(((cur[p - L:p] - D(p - L, p)) ** 2).sum(-1)); Dl = D(p - L, p)
            for r in range(h): cur[p - L + path[r]:p, r] = Dl[path[r]:, r]
            cur[p:p + m] = D(p, p + m)
            if last:
                tg = cur[b + 1:b + 1 + L].copy(); Dr = D(b + 1, b + 1 + L); path = seam_path(((tg - Dr) ** 2).sum(-1))
                for r in range(h): cur[b + 1:b + 1 + path[r], r] = Dr[:path[r], r]
            p += m
    return np.roll(cur, c0, axis=0), np.roll(cD, c0), log


def main():
    sc = Scene(sys.argv[1]); out = os.path.join(sc.work, 'plates'); os.makedirs(out, exist_ok=True)
    for p in sc.spec.get('effects', []):
        if p['type'] != 'rotate_ring' or p.get('off'): continue
        rid = str(p.get('id', p['r_in'])); cx, cy = sc.U(*p['center']); f = sc.f
        a, b = int(round(p['r_in'] * f)), int(round(p['r_out'] * f)); RB = b + 12
        L = p.get('seam', 50)
        pol = lambda img, it=cv2.INTER_LINEAR: cv2.warpPolar(img, (RB, TH), (cx, cy), RB, it + cv2.WARP_POLAR_LINEAR)
        occ = sc.matte_union(p.get('occluders', []), 9) if p.get('occluders') else np.zeros((sc.H, sc.W), np.float32)
        for x, y, r in p.get('occluder_circles', []):
            cv2.circle(occ, (int(x * sc.sx), int(y * sc.sy)), int(r * f), 1, -1)
        P = pol(sc.upf); Occ = pol((occ * 255).astype(np.uint8)) > 100
        bad = np.zeros(TH, bool)
        for d0, d1 in p.get('bad_deg', []): bad[int(d0 * TH / 360) % TH:int(d1 * TH / 360) + 1] = True
        try:
            filled, hc, log = fill_band(P[:, a:b], Occ[:, a:b], bad, L=L)
        except RuntimeError as e:
            print('ring', rid, ':', e, '-> using the original'); cv2.imwrite(os.path.join(out, rid + '.png'), sc.up); continue
        print('ring', rid, 'rows', a, b, 'hole columns', int(hc.sum()), 'segments', len(log))
        full = P.copy(); full[:, a:b] = filled
        hole = np.zeros((TH, RB), np.float32)
        ext = np.convolve(np.r_[hc[-L:], hc, hc[:L]].astype(int), np.ones(2 * L + 1, int), 'valid') > 0   # widen by the overlap
        hole[:, a:b] = ext[:, None]
        inv = lambda img: cv2.warpPolar(img, (sc.W, sc.H), (cx, cy), RB, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
        back = inv(full.astype(np.float32)); hm = cv2.GaussianBlur(inv(hole), (0, 0), 1.5)[..., None]
        plate = sc.upf * (1 - hm) + back * hm
        cv2.imwrite(os.path.join(out, rid + '.png'), np.clip(plate, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out, rid + '_polar_before_after.png'),
                    np.vstack([cv2.resize(np.transpose(P[:, a:b], (1, 0, 2))[::-1].astype(np.uint8), (1800, 160)),
                               cv2.resize(np.transpose(full[:, a:b], (1, 0, 2))[::-1].astype(np.uint8), (1800, 160))]))


if __name__ == '__main__':
    main()
