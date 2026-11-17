"""Effect library. One class per effect; each implements any of these stage methods:

  bg(base, t)            modify the clean background plate          (sky drift, water, rotating rings, flowing texture)
  back(img, t)           draw behind the objects                    (far particles)
  obj(layers, t)         transform premultiplied object layers      (sway, bob, breathe)
  post(comp, t)          modify the finished composite              (hair/grass that is not in any matte)
  light(acc, t)          add light into an additive float buffer    (glows, rays, embers, glints)
  front(img, t)          draw in front of the objects               (near particles)
  final(img, t)          whole-frame look                           (bloom, vignette, grade, camera, frame lock)

Every motion is built from integer cycles per clip length T, so the clip loops seamlessly.
Colours in scene.json are RGB; positions are SOURCE pixels.
"""
import math
import cv2, numpy as np
from core import smooth, lownoise, warp_roi

TAU = 2 * math.pi
REG = {}


def effect(name):
    def deco(cls):
        REG[name] = cls
        cls.name = name
        return cls
    return deco


def bgr(c):
    return np.array([c[2], c[1], c[0]], np.float32)


def cyc(T, per_second):
    """Nearest integer number of cycles in T for a rate given in cycles/second (>=1)."""
    return max(1, int(round(per_second * T)))


class Effect:
    stage_prio = 50

    def __init__(self, sc, p):
        self.sc, self.p = sc, p


# =====================================================================================  BACKGROUND (bg stage)
@effect('sky_drift')
class SkyDrift(Effect):
    """Clouds / haze / sky drifting: smooth noise warp, weights fade at object, frame and rim edges."""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        w = sc.poly(p['poly'], 6) if 'poly' in p else np.ones((sc.H, sc.W), np.float32)
        if 'exclude_poly' in p:
            w = w * (1 - sc.poly(p['exclude_poly'], 6))
        if sc.frame is not None:
            w = w * (1 - sc.frame)
        names = p.get('protect')
        w = w * (1 - np.clip(cv2.GaussianBlur(sc.matte_union(names, 15), (0, 0), 4), 0, 1))
        self.w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 4)
        self.amp = p.get('amp', 5.0)
        cell = p.get('cell', 90)
        self.nA = lownoise(sc.H, sc.W, p.get('seed', 11), 25, cell)
        self.nB = lownoise(sc.H, sc.W, p.get('seed', 11) + 1, 25, cell)

    def bg(self, base, t):
        sc = self.sc; ph = TAU * t / sc.T
        c, s = math.cos(ph), math.sin(ph)
        dx = self.w * self.amp * (self.nA * c + self.nB * s)
        dy = self.w * self.amp * 0.6 * (self.nB * c - self.nA * s)
        return cv2.remap(base, sc.XX + dx, sc.YY + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@effect('water_ripple')
class WaterRipple(Effect):
    """Lake / sea / pond: perspective ripple (small and slow far away, big and wavy near the viewer)."""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        pts = np.array(p['poly'], np.float32)
        self.m = sc.poly(p['poly'], 5)
        self.y0 = max(int(pts[:, 1].min() * sc.sy) - 12, 0); self.y1 = min(int(pts[:, 1].max() * sc.sy) + 12, sc.H)
        top, bot = pts[:, 1].min() * sc.sy, pts[:, 1].max() * sc.sy
        Ls = np.clip((sc.YY - top) / (bot - top + 1e-6), 0, 1)
        k = p.get('perspective', 3.0)
        self.Lu = np.log1p(k * Ls) / math.log(1 + k); self.Ls = Ls
        self.waves = p.get('waves', 10); self.cyc = p.get('cycles', 7)
        self.a0, self.a1 = p.get('amp_far', 1.2), p.get('amp_near', 4.6)
        self.shim = p.get('shimmer', 0.16)
        r = np.random.RandomState(p.get('seed', 9)); n = p.get('glints', 100)
        x0, x1 = pts[:, 0].min(), pts[:, 0].max(); y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        self.gl = [(r.uniform(x0, x1) * sc.sx, r.uniform(y0, y1) * sc.sy, cyc(sc.T, r.uniform(0.07, 0.13)), r.uniform(0, TAU),
                    r.uniform(8, 26) * sc.f) for _ in range(n)]
        self.gl = [g for g in self.gl if self.m[min(int(g[1]), sc.H - 1), min(int(g[0]), sc.W - 1)] > 0.5]
        self.gcol = bgr(p.get('glint_color', [255, 235, 200]))

    def bg(self, base, t):
        sc = self.sc; a, b = self.y0, self.y1
        ph = TAU * (self.waves * self.Lu[a:b] - self.cyc * t / sc.T)
        amp = (self.a0 + (self.a1 - self.a0) * self.Ls[a:b]) * self.m[a:b]
        dx = amp * (np.sin(ph) + 0.4 * np.sin(2.3 * ph + 1.0)); dy = 0.35 * amp * np.sin(ph + 1.6)
        rip = cv2.remap(base[a:b], sc.XX[a:b] + dx, (sc.YY[a:b] - a) + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        shim = 1.0 + self.shim * self.m[a:b, :, None] * np.sin(ph[..., None] + 2.0) * (0.4 + self.Ls[a:b, :, None])
        base[a:b] = rip * shim
        return base

    def light(self, acc, t):
        sc = self.sc
        for x, y, c, ph, ln in self.gl:
            v = max(0.0, math.sin(TAU * c * t / sc.T + ph)) ** 3
            if v > 0.05:
                cv2.line(acc, (int(x - ln / 2), int(y)), (int(x + ln / 2), int(y)),
                         tuple(float(k) for k in self.gcol * v * 0.7), 1, cv2.LINE_AA)


@effect('flow')
class Flow(Effect):
    """Waterfall / stream / mist / smoke: scrolling blurred-noise texture, optionally only where the pixels are
    bright & unsaturated (white water)."""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        pts = np.array(p['poly'], np.float32)
        self.x0, self.x1 = int(pts[:, 0].min() * sc.sx), int(pts[:, 0].max() * sc.sx)
        self.y0, self.y1 = int(pts[:, 1].min() * sc.sy), int(pts[:, 1].max() * sc.sy)
        pm = sc.poly(p['poly'])
        if p.get('mode', 'bright') == 'bright':
            hsv = cv2.cvtColor(sc.up, cv2.COLOR_BGR2HSV)
            pm = pm * ((hsv[..., 2] > p.get('vmin', 120)) & (hsv[..., 1] < p.get('smax', 120)))
        m = cv2.morphologyEx(pm.astype(np.float32), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.m = cv2.GaussianBlur(m, (0, 0), 2)[self.y0:self.y1, self.x0:self.x1, None]
        self.dir = p.get('direction', 'down'); self.vert = self.dir in ('down', 'up')
        h, w = (self.y1 - self.y0, self.x1 - self.x0) if self.vert else (self.x1 - self.x0, self.y1 - self.y0)
        self.L = max(1400, 2 * h)   # texture length along the motion axis
        mk = lambda seed, sa, sb: self._tex(seed, w, sa, sb)
        self.t1, self.t2 = mk(3, 1.6, 14), mk(4, 2.5, 22)
        sp = p.get('speed', 300) * sc.f
        self.c1 = cyc(sc.T, sp / self.L); self.c2 = cyc(sc.T, 0.63 * sp / self.L)
        self.hl = p.get('highlight', 150); self.st = p.get('strength', 0.9)

    def _tex(self, seed, w, sa, sb):
        t = np.random.RandomState(seed).rand(self.L, w).astype(np.float32)
        t = cv2.GaussianBlur(t, (0, 0), sigmaX=sa, sigmaY=sb)
        return (t - t.min()) / (t.max() - t.min())

    def bg(self, base, t):
        sc = self.sc; a, b, c, d = self.y0, self.y1, self.x0, self.x1
        sgn = 1 if self.dir in ('down', 'right') else -1
        r1 = np.roll(self.t1, sgn * int(self.L * self.c1 * t / sc.T), axis=0)
        r2 = np.roll(self.t2, sgn * int(self.L * self.c2 * t / sc.T), axis=0)
        fl = 0.6 * r1 + 0.4 * r2
        n = (b - a) if self.vert else (d - c)
        fl = fl[:n]
        if not self.vert: fl = fl.T
        fl = fl[..., None]
        cur = base[a:b, c:d]
        base[a:b, c:d] = np.clip(cur * (1 + self.m * (fl - 0.5) * self.st) + self.m * np.maximum(fl - 0.62, 0) * self.hl, 0, 255)
        return base


@effect('rotate_ring')
class RotateRing(Effect):
    """Rotate an annulus (mandala, halo, wheel, clock face, planet ring) about a centre.
    The plate (tools/polar_fill.py) already contains the hidden parts, so objects in front stay put.
    deg_per_sec: negative = clockwise on screen. ease_tau: seconds of gentle start (0 = constant speed)."""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        cx, cy = sc.U(*p['center']); self.C = (float(cx), float(cy))
        r0, r1, fe = p['r_in'] * sc.f, p['r_out'] * sc.f, p.get('feather', 1.5)
        pad = int(r1 + 8)
        self.x0, self.x1 = max(int(cx) - pad, 0), min(int(cx) + pad, sc.W)
        self.y0, self.y1 = max(int(cy) - pad, 0), min(int(cy) + pad, sc.H)
        rr = np.hypot(sc.XX[self.y0:self.y1, self.x0:self.x1] - cx, sc.YY[self.y0:self.y1, self.x0:self.x1] - cy)
        self.A = (np.clip((rr - (r0 - fe)) / (2 * fe), 0, 1) * np.clip(((r1 + fe) - rr) / (2 * fe), 0, 1))[..., None].astype(np.float32)
        pl = p.get('plate', 'plates/%s.png' % p.get('id', p['r_in']))
        import os
        pp = os.path.join(sc.work, pl)
        if os.path.exists(pp):
            plate = cv2.imread(pp).astype(np.float32)
        else:
            plate = sc.upf
            print('! no plate for ring %s: hidden parts will rotate with it (run tools/polar_fill.py)' % pl)
        self.plate = plate[self.y0:self.y1, self.x0:self.x1].copy()
        self.dps, self.tau = p['deg_per_sec'], p.get('ease_tau', 1.2)

    def ang(self, t):
        if self.tau <= 0: return self.dps * t
        return self.dps * (t - self.tau * (1 - math.exp(-t / self.tau)))

    def bg(self, base, t):
        c = (self.C[0] - self.x0, self.C[1] - self.y0)
        M = cv2.getRotationMatrix2D(c, self.ang(t), 1.0)
        w = cv2.warpAffine(self.plate, M, (self.x1 - self.x0, self.y1 - self.y0), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        roi = base[self.y0:self.y1, self.x0:self.x1]
        base[self.y0:self.y1, self.x0:self.x1] = roi * (1 - self.A) + w * self.A
        return base


# =====================================================================================  OBJECT MOTION
@effect('sway')
class Sway(Effect):
    """Wind on cloth / ribbon / flag / hair / grass / tree / curtain.
    target 'object'  : displaces the premultiplied layer of one object (silhouette really moves, reveals clean bg).
    target 'composite': displaces the finished picture (for things that are not in any matte, e.g. hair, grass).
    Weight = (distance from anchor / reach)^power  x  optional half-plane  x  polygon  x  not protected."""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        self.comp = p.get('target', 'object') == 'composite'
        ax, ay = sc.U(*p['anchor']); reach = p['reach'] * sc.f
        X, Y = sc.XX, sc.YY; mode = p.get('dist', 'radial')
        d = np.hypot(X - ax, Y - ay) if mode == 'radial' else (np.abs(X - ax) if mode == 'x' else np.abs(Y - ay))
        w = np.clip(d / reach, 0, 1) ** p.get('power', 1.0)
        side, soft = p.get('side'), max(p.get('soft', 16) * sc.f, 1)
        if side == 'right': w = w * np.clip((X - ax) / soft, 0, 1)
        if side == 'left': w = w * np.clip((ax - X) / soft, 0, 1)
        if side == 'below': w = w * np.clip((Y - ay) / soft, 0, 1)
        if side == 'above': w = w * np.clip((ay - Y) / soft, 0, 1)
        if 'poly' in p: w = w * sc.poly(p['poly'], 10)
        for pp in p.get('protect_polys', []): w = w * (1 - sc.poly(pp, 8))
        for pp in p.get('keep_still_x', []):   # [x_center, half_width] in src px: vertical strip kept still (e.g. a leg, a pole)
            w = w * smooth(np.abs(X - pp[0] * sc.sx), pp[1] * sc.sx * 0.55, pp[1] * sc.sx)
        if self.comp and p.get('protect'):
            w = w * (1 - cv2.GaussianBlur(sc.matte_union(p['protect'], 9), (0, 0), 3))
        w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 8) * p.get('gain', 1.0)
        ys, xs = np.where(w > 0.01)
        if len(ys) == 0: raise SystemExit('sway: empty weight map for ' + str(p.get('object')))
        pad = int(max(p['amp']) * sc.f) + 6
        self.x0, self.x1 = max(xs.min() - pad, 0), min(xs.max() + pad, sc.W)
        self.y0, self.y1 = max(ys.min() - pad, 0), min(ys.max() + pad, sc.H)
        self.w = w[self.y0:self.y1, self.x0:self.x1]
        self.X = sc.XX[self.y0:self.y1, self.x0:self.x1] / sc.sx; self.Y = sc.YY[self.y0:self.y1, self.x0:self.x1] / sc.sy
        self.cyc = p.get('cycles', 6); self.lam = p.get('wavelength', 95); self.k = p.get('wave_dir', [1.0, 0.5])
        self.amp = [a * sc.f for a in p['amp']]; self.ph0 = p.get('phase', 0.0)

    def _disp(self, t):
        sc = self.sc
        ph = TAU * (self.cyc * t / sc.T - (self.k[0] * self.X + self.k[1] * self.Y) / self.lam + self.ph0)
        dx = self.amp[0] * self.w * (np.sin(ph) + 0.5 * np.sin(2 * ph + 1.0))
        dy = self.amp[1] * self.w * np.sin(ph + 1.9)
        return dx.astype(np.float32), dy.astype(np.float32)

    def obj(self, layers, t):
        if self.comp: return
        n = self.p['object']
        if n not in layers: return
        dx, dy = self._disp(t)
        L = layers[n].copy()
        L[self.y0:self.y1, self.x0:self.x1] = warp_roi(layers[n], self.x0, self.y0, self.x1, self.y1, dx, dy)
        layers[n] = L

    def post(self, comp, t):
        if not self.comp: return comp
        dx, dy = self._disp(t)
        comp[self.y0:self.y1, self.x0:self.x1] = warp_roi(comp, self.x0, self.y0, self.x1, self.y1, dx, dy)
        return comp


@effect('bob')
class Bob(Effect):
    """Rigid motion of one object: floating / hovering / swinging lantern / bobbing boat / breathing.
    move [ax, ay] src px; rotate degrees about pivot; scale amplitude (0.004 = chest breathing); cycles per clip."""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        self.piv = sc.U(*p.get('pivot', [sc.sw / 2, sc.sh / 2]))
        self.mv = [m * sc.f for m in p.get('move', [0, 0])]
        self.rot, self.sca = p.get('rotate', 0.0), p.get('scale', 0.0)
        self.c = p.get('cycles', 2); self.ph = p.get('phase', 0.0)

    def obj(self, layers, t):
        n = self.p['object']
        if n not in layers: return
        sc = self.sc; s = math.sin(TAU * (self.c * t / sc.T + self.ph))
        M = cv2.getRotationMatrix2D((float(self.piv[0]), float(self.piv[1])), self.rot * s, 1.0 + self.sca * s)
        M[0, 2] += self.mv[0] * s; M[1, 2] += self.mv[1] * math.sin(TAU * (self.c * t / sc.T + self.ph + 0.25))
        layers[n] = cv2.warpAffine(layers[n], M, (sc.W, sc.H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


# =====================================================================================  LIGHT (additive)
@effect('glow')
class Glow(Effect):
    """Glowing eye, diya flame, lamp, star, jewel, window, halo, sun. anim: const | pulse | flicker | flash | ramp | blink.
    pos = [x,y] or points = [[x,y],..]; sigma in src px; rx/ry makes it an ellipse; t0/t1 time window for ramp/flash."""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        self.pts = [sc.U(*q) for q in (p['points'] if 'points' in p else [p['pos']])]
        self.col = bgr(p.get('color', [255, 190, 90]))
        self.sig = p.get('sigma', 30) * sc.f
        self.rx, self.ry = p.get('rx'), p.get('ry')
        self.I = p.get('intensity', 0.5); self.anim = p.get('anim', 'const')
        self.cycles = cyc(sc.T, 1.0 / p.get('period', 2.8))
        self.t0, self.t1, self.wd = p.get('t0', 0.0), p.get('t1', 2.0), p.get('width', 0.45)
        self.floor = p.get('floor', 0.0); self.ph = [np.random.RandomState(p.get('seed', 1) + i).uniform(0, TAU) for i in range(len(self.pts))]

    def k(self, t, i):
        T = self.sc.T; a = self.anim
        if a == 'pulse': v = 0.8 + 0.2 * math.sin(TAU * self.cycles * t / T + self.ph[i])
        elif a == 'flicker':
            v = 0.72 + 0.16 * math.sin(TAU * 25 * t / T + self.ph[i]) + 0.12 * math.sin(TAU * 51 * t / T + 2 * self.ph[i])
        elif a == 'flash': v = math.exp(-((t - self.t0) / self.wd) ** 2)
        elif a == 'ramp': v = float(smooth(np.array([t]), self.t0, self.t1)[0])
        elif a == 'blink': v = max(0.0, math.sin(TAU * self.cycles * t / T + self.ph[i])) ** 2
        else: v = 1.0
        return self.floor + (1 - self.floor) * v

    def light(self, acc, t):
        sc = self.sc
        for i, (cx, cy) in enumerate(self.pts):
            I = self.I * self.k(t, i)
            if I < 0.004: continue
            rx = (self.rx or 0) * sc.f; ry = (self.ry or 0) * sc.f
            sx_, sy_ = (rx, ry) if rx and ry else (self.sig, self.sig)
            r = int(max(sx_, sy_) * 3.2)
            x0, x1, y0, y1 = max(int(cx) - r, 0), min(int(cx) + r, sc.W), max(int(cy) - r, 0), min(int(cy) + r, sc.H)
            if x1 <= x0 or y1 <= y0: continue
            xs = np.arange(x0, x1, dtype=np.float32)[None, :] - cx; ys = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
            g = np.exp(-((xs / sx_) ** 2 + (ys / sy_) ** 2) / 2) * I
            acc[y0:y1, x0:x1] += g[..., None] * self.col


@effect('rays')
class Rays(Effect):
    """Rotating god-rays / sun-burst around a centre, additive. sets = [[n_rays, rev_per_sec, phase, weight], ...]"""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        cx, cy = sc.U(*p['center']); r1 = p['r1'] * sc.f; pad = int(r1 + 4)
        self.x0, self.x1 = max(int(cx) - pad, 0), min(int(cx) + pad, sc.W)
        self.y0, self.y1 = max(int(cy) - pad, 0), min(int(cy) + pad, sc.H)
        X = sc.XX[self.y0:self.y1, self.x0:self.x1] - cx; Y = sc.YY[self.y0:self.y1, self.x0:self.x1] - cy
        rr = np.hypot(X, Y); self.th = np.arctan2(Y, X)
        a, b = p.get('r0', 40) * sc.f, r1
        self.rw = (smooth(rr, a, a + (b - a) * 0.25) * (1 - smooth(rr, a + (b - a) * 0.7, b))).astype(np.float32)
        self.sets = p.get('sets', [[7, 0.02, 0.3, 1.0], [11, -0.013, 1.7, 0.8]])
        self.I = p.get('intensity', 0.16); self.col = bgr(p.get('color', [255, 210, 120]))
        prot = p.get('protect')
        self.free = (1 - np.clip(sc.matte_union(prot)[self.y0:self.y1, self.x0:self.x1] * 0.8, 0, 1))[..., None] if prot is not None else 1.0
        self.pw = p.get('sharpness', 6)

    def light(self, acc, t):
        sc = self.sc; s = np.zeros_like(self.rw)
        for n, sp, ph, wt in self.sets:
            rev = max(1, round(abs(sp) * sc.T)) * (1 if sp >= 0 else -1) / sc.T if sp else 0   # integer revolutions per clip
            s += wt * np.maximum(0, np.cos(n * self.th - TAU * rev * t + ph)) ** self.pw
        pul = 0.8 + 0.2 * math.sin(TAU * cyc(sc.T, 1 / 2.8) * t / sc.T)
        acc[self.y0:self.y1, self.x0:self.x1] += (s * self.rw * pul * self.I)[..., None] * self.col * self.free


# =====================================================================================  PARTICLES
def _petal(palette, size=96, ss=3, wide=0.36):
    N = size * ss
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    u = (yy - N * 0.05) / (N * 0.9); v = (xx - N / 2) / (N * wide)
    prof = np.where((u > 0) & (u < 1), np.clip(np.sin(np.pi * np.clip(u, 0, 1) ** 0.7), 0, None) ** 0.85, 0)
    m = cv2.GaussianBlur((np.abs(v) < prof).astype(np.float32), (0, 0), 1.2 * ss / 2)
    t = np.clip(u, 0, 1)[..., None]
    tip, mid, base = [np.array(c, np.float32)[None, None, :] for c in palette]
    col = np.where(t < 0.5, tip + (mid - tip) * (t / 0.5), mid + (base - mid) * ((t - 0.5) / 0.5))
    vein = np.exp(-(v * 6) ** 2) * 0.18
    col = col * (1 - vein[..., None]) + 255 * vein[..., None] * 0.4
    cv = np.zeros((N, N, 4), np.float32); cv[..., :3] = col[..., ::-1] / 255.; cv[..., 3] = m
    return cv


def _disc(size=64, hard=0.35):
    r = np.hypot(*(np.mgrid[0:size, 0:size].astype(np.float32) - size / 2)) / (size / 2)
    a = np.clip((1 - r) / (1 - hard + 1e-6), 0, 1) ** 1.5
    cv = np.ones((size, size, 4), np.float32); cv[..., 3] = a
    return cv


def _star(size=64):
    c = size / 2; yy, xx = np.mgrid[0:size, 0:size].astype(np.float32); x, y = xx - c, yy - c
    a = np.maximum(np.exp(-(x / 2.2) ** 2 - (y / (c * 0.9)) ** 2), np.exp(-(y / 2.2) ** 2 - (x / (c * 0.9)) ** 2))
    a = np.maximum(a, np.exp(-(x * x + y * y) / (2 * 4.0 ** 2)) * 0.9)
    cv = np.ones((size, size, 4), np.float32); cv[..., 3] = np.clip(a, 0, 1)
    return cv


def _leaf(green=True, size=96, ss=3):
    N = size * ss; img = np.zeros((N, N, 3), np.float32); a = np.zeros((N, N), np.float32)
    col = (60, 150, 70) if green else (40, 110, 220)
    cv2.ellipse(a, (N // 2, N // 2), (int(N * 0.42), int(N * 0.17)), 90, 0, 360, 1, -1)
    img[:] = col; cv2.line(img, (N // 2, int(N * 0.08)), (N // 2, int(N * 0.92)), tuple(int(c * 0.6) for c in col), ss)
    a = cv2.GaussianBlur(a, (0, 0), ss / 2); cv = np.zeros((N, N, 4), np.float32); cv[..., :3] = img / 255.; cv[..., 3] = a
    return cv


def make_sprite(kind, size=96):
    pal = dict(lotus=((255, 205, 228), (245, 110, 170), (190, 40, 110)),
               sakura=((255, 240, 245), (255, 190, 210), (240, 140, 175)),
               rose=((250, 120, 120), (200, 30, 50), (120, 10, 30)),
               hibiscus=((255, 160, 120), (240, 60, 60), (150, 20, 40)))
    if kind in pal:
        cv = _petal(pal[kind], size, 3, 0.36 if kind != 'sakura' else 0.42)
    elif kind == 'marigold':
        N = size * 3; c = N / 2; img = np.zeros((N, N, 3), np.float32); a = np.zeros((N, N), np.float32)
        for ring, (rad, col, n) in enumerate([(0.44, (40, 150, 250), 14), (0.34, (25, 175, 255), 12), (0.22, (10, 120, 230), 9)]):
            for k in range(n):
                ang = TAU * k / n + ring * 0.3
                m = np.zeros((N, N), np.uint8)
                cv2.ellipse(m, (int(c + np.cos(ang) * rad * N * 0.5), int(c + np.sin(ang) * rad * N * 0.5)), (int(N * 0.13), int(N * 0.075)),
                            np.degrees(ang), 0, 360, 255, -1)
                img[m > 0] = col; a[m > 0] = 1
        cv2.circle(img, (int(c), int(c)), int(N * 0.07), (0, 90, 200), -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.5); a = cv2.GaussianBlur(a, (0, 0), 1.5)
        cv = np.zeros((N, N, 4), np.float32); cv[..., :3] = img / 255.; cv[..., 3] = a
    elif kind == 'jasmine':
        N = size * 3; c = N / 2; img = np.zeros((N, N, 3), np.float32); a = np.zeros((N, N), np.float32)
        for k in range(5):
            ang = TAU * k / 5; m = np.zeros((N, N), np.uint8)
            cv2.ellipse(m, (int(c + np.cos(ang) * N * 0.2), int(c + np.sin(ang) * N * 0.2)), (int(N * 0.2), int(N * 0.1)), np.degrees(ang), 0, 360, 255, -1)
            img[m > 0] = (235, 245, 255); a[m > 0] = 1
        cv2.circle(img, (int(c), int(c)), int(N * 0.06), (60, 200, 255), -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.5); a = cv2.GaussianBlur(a, (0, 0), 1.5)
        cv = np.zeros((N, N, 4), np.float32); cv[..., :3] = img / 255.; cv[..., 3] = a
    elif kind == 'leaf': cv = _leaf(True, size)
    elif kind == 'autumn': cv = _leaf(False, size)
    elif kind == 'star': cv = _star(size)
    else: cv = _disc(size)
    cv = cv2.resize(cv, (size, size), interpolation=cv2.INTER_AREA); cv[..., :3] *= cv[..., 3:4]
    return cv


ADD_KINDS = {'ember', 'firefly', 'sparkle', 'dust', 'star'}


@effect('particles')
class Particles(Effect):
    """Falling / rising things. kinds: lotus sakura rose hibiscus marigold jasmine leaf autumn (alpha-blended petals & leaves),
    snow, rain (alpha), ember firefly sparkle dust (additive glow).  layer: 'back' (behind objects) | 'front' | 'light'.
    All loops are integer-periodic."""
    def __init__(self, sc, p):
        super().__init__(sc, p)
        self.layer = p.get('layer', 'front'); r = np.random.RandomState(p.get('seed', 21))
        kinds = p.get('kinds', ['lotus']); n = p.get('count', 30); smin, smax = p.get('size', [0.5, 1.0])
        sp0, sp1 = p.get('speed', [95, 190]); direction = p.get('direction', 'down'); self.dir = direction
        m = 130; self.span = sc.H + 2 * m; self.m = m; Wm = sc.W + 2 * m; self.Wm = Wm
        self.spr = {k: make_sprite(k) for k in set(kinds)}
        self.sprb = {k: cv2.GaussianBlur(v, (0, 0), 1.8) for k, v in self.spr.items()}
        self.col = bgr(p.get('color', [255, 190, 90])); self.add = p.get('blend') == 'add' or kinds[0] in ADD_KINDS
        self.alpha = p.get('alpha', 0.93); self.rain = kinds[0] == 'rain'; self.blur_thr = p.get('blur_above', 1.15)
        self.P = []
        for i in range(n):
            s = r.uniform(smin, smax); vy = r.uniform(sp0, sp1) * (0.6 + 0.5 * s if p.get('size_speed', True) else 1.0) * sc.f
            vy = cyc(sc.T, vy / self.span) * self.span / sc.T
            vx = r.uniform(*p.get('wind', [20, 70])) * (0.6 + 0.5 * s) * sc.f * r.choice([1, 1, 1, -1] if p.get('wind_both') else [1])
            vx = round(vx * sc.T / Wm) * Wm / sc.T
            rv = r.uniform(-90, 90); rv = round(rv * sc.T / 360) * 360 / sc.T
            self.P.append(dict(k=kinds[r.randint(len(kinds))], s=s, x0=r.uniform(-m, sc.W + m), y0=r.uniform(0, 1), vy=vy, vx=vx,
                               ax=r.uniform(25, 70) * (0.5 + 0.5 * s) * sc.f, cx=cyc(sc.T, r.uniform(0.04, 0.09)), ph=r.uniform(0, TAU),
                               rot0=r.uniform(0, 360), rv=rv, ct=cyc(sc.T, r.uniform(0.09, 0.2)), tp=r.uniform(0, TAU),
                               tw=cyc(sc.T, r.uniform(0.15, 0.45))))
        self.unit = p.get('unit', 0.5)

    def _pos(self, q, t):
        sc = self.sc
        if self.dir == 'up': y = sc.H + self.m - ((q['y0'] * self.span + q['vy'] * t) % self.span)
        else: y = ((q['y0'] * self.span + q['vy'] * t) % self.span) - self.m
        x = q['x0'] + q['vx'] * t + q['ax'] * math.sin(TAU * q['cx'] * t / sc.T + q['ph'])
        return ((x + self.m) % self.Wm) - self.m, y

    PETAL = {'lotus': .55, 'marigold': .48, 'jasmine': .32, 'sakura': .5, 'rose': .5, 'hibiscus': .5, 'leaf': .5, 'autumn': .5}

    def _draw(self, img, t):
        sc = self.sc; uf = sc.W / 1080.0
        for q in self.P:
            x, y = self._pos(q, t)
            if self.rain:
                L = 28 * q['s'] * uf
                c = tuple(float(v) for v in (self.col if self.p.get('color') else np.array([235, 225, 210])) * q['s'] * 0.8)
                cv2.line(img, (int(x), int(y)), (int(x - L * 0.18), int(y - L)), c, 1, cv2.LINE_AA)
                continue
            k = q['k']; sprite = (self.sprb if q['s'] > self.blur_thr else self.spr)[k]
            if k in self.PETAL:      # tumbling petal / leaf
                tum = math.cos(TAU * q['ct'] * t / sc.T + q['tp'])
                ang = q['rot0'] + q['rv'] * t + 25 * math.sin(TAU * q['cx'] * t / sc.T + q['ph'])
                s_ = q['s'] * self.PETAL[k] * uf
                self._sprite(img, sprite, x, y, ang, s_ * (0.35 + 0.65 * abs(tum)), s_, self.alpha)
            else:                    # round things: snow, dust, ember, firefly, sparkle
                tw = 0.45 + 0.55 * max(0, math.sin(TAU * q['tw'] * t / sc.T + q['ph'])) ** 2 if self.add else 1.0
                s_ = q['s'] * self.unit * 0.55 * uf
                self._sprite(img, sprite, x, y, 0, s_, s_, self.alpha * tw)

    def _sprite(self, img, spr, cx, cy, ang, sxs, sys_, alpha):
        h, w = spr.shape[:2]; sc = self.sc
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0); M[0, :2] *= sxs; M[1, :2] *= sys_
        R = int(max(w, h) * max(sxs, sys_) * 0.75) + 2
        M[0, 2] = R - (M[0, 0] * w / 2 + M[0, 1] * h / 2); M[1, 2] = R - (M[1, 0] * w / 2 + M[1, 1] * h / 2)
        patch = cv2.warpAffine(spr, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        x0, y0 = int(cx) - R, int(cy) - R
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + 2 * R, sc.W), min(y0 + 2 * R, sc.H)
        if X1 <= X0 or Y1 <= Y0: return
        pt = patch[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]; a = pt[..., 3:4] * alpha
        if self.add:
            img[Y0:Y1, X0:X1] += a * self.col
        else:
            col = pt[..., :3] * 255 if self.p.get('color') is None else self.col * pt[..., 3:4]
            img[Y0:Y1, X0:X1] = img[Y0:Y1, X0:X1] * (1 - a) + col * alpha

    def back(self, img, t):
        if self.layer == 'back': self._draw(img, t)

    def front(self, img, t):
        if self.layer == 'front': self._draw(img, t)

    def light(self, acc, t):
        if self.layer == 'light': self._draw(acc, t)


# =====================================================================================  FINAL LOOK
@effect('bloom')
class Bloom(Effect):
    stage_prio = 10
    def final(self, img, t):
        p = self.p; thr = p.get('threshold', 190); sg = p.get('sigma', 7); k = p.get('strength', 0.16)
        small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        lum = small.mean(2, keepdims=True)
        b = cv2.GaussianBlur(small * (lum > thr), (0, 0), sg)
        return img + cv2.resize(b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) * k


@effect('vignette')
class Vignette(Effect):
    stage_prio = 20
    def __init__(self, sc, p):
        super().__init__(sc, p)
        r = np.hypot((sc.XX - sc.W / 2) / (sc.W / 2), (sc.YY - sc.H / 2) / (sc.H / 2)) / 1.4
        self.v = (1 - p.get('strength', 0.25) * smooth(r, 0.5, 1.2))[..., None].astype(np.float32)
    def final(self, img, t): return img * self.v


@effect('grade')
class Grade(Effect):
    """warm/cool tint (RGB gains), contrast, saturation. Cheap colour look."""
    stage_prio = 30
    def final(self, img, t):
        p = self.p; g = bgr(p.get('gain', [1, 1, 1])); c = p.get('contrast', 1.0); s = p.get('saturation', 1.0)
        img = (img * g - 128) * c + 128
        if s != 1.0:
            gray = img.mean(2, keepdims=True); img = gray + (img - gray) * s
        return img


@effect('camera')
class Camera(Effect):
    """Slow push-in / pull-out / drift (Ken Burns). loop=true goes out and back so the clip loops."""
    stage_prio = 40
    def final(self, img, t):
        sc = self.sc; p = self.p; u = t / sc.T
        e = (0.5 - 0.5 * math.cos(TAU * u)) if p.get('loop') else (0.5 - 0.5 * math.cos(math.pi * u))
        z = p.get('zoom0', 1.0) + (p.get('zoom1', 1.03) - p.get('zoom0', 1.0)) * e
        c = p.get('center', [sc.sw / 2, sc.sh / 2]); cx, cy = sc.U(*c)
        pan = p.get('pan', [0, 0])
        M = np.array([[z, 0, cx - z * cx + pan[0] * sc.sx * e], [0, z, cy - z * cy + pan[1] * sc.sy * e]], np.float32)
        return cv2.warpAffine(img, M, (sc.W, sc.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@effect('frame_lock')
class FrameLock(Effect):
    """Keep a border / ornament / logo / caption area perfectly still: pasted from the original after everything else."""
    stage_prio = 90
    def final(self, img, t):
        fr = self.sc.frame
        if fr is None: return img
        return img * (1 - fr[..., None]) + self.sc.upf * fr[..., None]
