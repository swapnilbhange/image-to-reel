"""Scene loader + small helpers shared by every effect.

All coordinates inside scene.json are in SOURCE-image pixels. Scene.U() / Scene.poly() convert them to
output pixels, so you can read positions straight off the grid sheets made by tools/inspect.py.
"""
import json, os, math
import cv2, numpy as np


def smooth(x, a, b):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def lownoise(h, w, seed, blur=25, cell=90):
    """Smooth random field in [-1,1] at size (h,w)."""
    r = np.random.RandomState(seed).randn(max(h // cell, 2), max(w // cell, 2)).astype(np.float32)
    r = cv2.resize(r, (w, h), interpolation=cv2.INTER_CUBIC)
    r = cv2.GaussianBlur(r, (0, 0), blur)
    return r / (np.abs(r).max() + 1e-9)


def warp_roi(img, x0, y0, x1, y1, dx, dy):
    """Displace a sub-rectangle. dx, dy are ROI-sized. (ROI-relative grids: absolute ones garble the patch.)"""
    h, w = y1 - y0, x1 - x0
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(np.ascontiguousarray(img[y0:y1, x0:x1]), gx + dx, gy + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


class Scene:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dir = os.path.dirname(self.path)
        s = self.spec = json.load(open(path))
        src = cv2.imread(os.path.join(self.dir, s['image']))
        if src is None:
            raise SystemExit('cannot read image ' + s['image'])
        self.src = src
        self.sh, self.sw = src.shape[:2]
        self.W, self.H = s.get('out_size', [1080, 1920])
        self.fps = s.get('fps', 30)
        self.T = float(s.get('duration', 14))
        self.nframes = int(round(self.fps * self.T))
        self.sx, self.sy = self.W / self.sw, self.H / self.sh
        self.f = (self.sx + self.sy) / 2          # isotropic scale for radii / sizes
        self.up = cv2.resize(src, (self.W, self.H), interpolation=cv2.INTER_LANCZOS4)
        self.upf = self.up.astype(np.float32)
        self.XX, self.YY = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        self.work = os.path.join(self.dir, s.get('work_dir', 'work'))
        # ---- objects (mattes made by tools/segment.py)
        self.objects = {}
        for i, o in enumerate(s.get('objects', [])):
            p = os.path.join(self.work, 'mattes', o['name'] + '.npy')
            if not os.path.exists(p):
                print('! matte missing for', o['name'], '(run tools/segment.py)'); continue
            m = np.load(p).astype(np.float32)
            if m.shape != (self.H, self.W):
                m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            m = np.clip(m, 0, 1)
            layer = np.dstack([self.upf * m[..., None], m]).astype(np.float32)   # premultiplied RGB + alpha
            self.objects[o['name']] = dict(spec=o, matte=m, layer=layer, z=o.get('z', i))
        # ---- clean background plate (objects lifted out), else the original
        bgp = os.path.join(self.work, 'bg.png')
        self.bg = cv2.imread(bgp).astype(np.float32) if os.path.exists(bgp) else self.upf.copy()
        # ---- frame lock mask
        fp = os.path.join(self.work, 'frame.npy')
        self.frame = None
        if os.path.exists(fp):
            fr = np.load(fp).astype(np.float32)
            self.frame = cv2.resize(fr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

    # ---------------- coordinate helpers
    def U(self, x, y):
        return x * self.sx, y * self.sy

    def poly(self, pts, blur=0.0):
        m = np.zeros((self.H, self.W), np.float32)
        cv2.fillPoly(m, [np.array([(int(x * self.sx), int(y * self.sy)) for x, y in pts], np.int32)], 1.0)
        return cv2.GaussianBlur(m, (0, 0), blur) if blur > 0 else m

    def matte_union(self, names=None, dilate=0):
        m = np.zeros((self.H, self.W), np.float32)
        for n, o in self.objects.items():
            if names is None or n in names:
                m = np.maximum(m, o['matte'])
        if dilate:
            m = cv2.dilate((m > 0.3).astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(np.float32)
        return m
