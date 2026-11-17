#!/usr/bin/env python3
"""Make the mattes (who stays in front), the clean background plate and the frame-lock mask from scene.json.

  python3 tools/segment.py scene.json                 # all objects + bg + frame
  python3 tools/segment.py scene.json figure cloth    # only these objects (then bg is rebuilt)
  python3 tools/segment.py scene.json --bg            # only rebuild bg.png / frame.npy

Per-object `method`:
  sam      strips:[{y0,y1,pos:[[x,y]..],neg:[[x,y]..]}]  Segment-Anything point prompts, one call per strip (src px)
  grabcut  box:[x0,y0,x1,y1], iters, fg:[[x,y]..], bg:[[x,y]..]
  color    poly:[[x,y]..], hsv_lo:[h,s,v], hsv_hi:[h,s,v]       pixels in a polygon whose colour is in range
  polygon  poly:[[x,y]..]                                       hand-drawn silhouette
  rembg    model:"isnet-general-use"                            whole-subject cut-out (portrait / product on plain bg)
  file     path:"matte.png"                                     a matte you made elsewhere
Clean-up (all optional, applied in this order): add_poly, cut_rects, close, fill_holes, specks, global_close, local_close,
guided, contrast. Everything is in SOURCE pixels; mattes are saved at output size as work/mattes/<name>.npy.
"""
import sys, os, json, subprocess
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    spec = json.load(open(path)); d = os.path.dirname(os.path.abspath(path))
    return spec, d, os.path.join(d, spec.get('work_dir', 'work'))


def ell(k): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# ------------------------------------------------------------------ SAM (one strip per process: the session leaks memory)
def sam_worker(scene, name, idx):
    from rembg import remove, new_session
    from PIL import Image
    spec, d, work = load(scene)
    o = [x for x in spec['objects'] if x['name'] == name][0]; st = o['strips'][idx]
    src = cv2.imread(os.path.join(d, spec['image'])); y0, y1 = st.get('y0', 0), st.get('y1', src.shape[0])
    crop = src[y0:y1]
    prompt = [{"type": "point", "label": 1, "data": [x, y - y0]} for x, y in st.get('pos', [])] + \
             [{"type": "point", "label": 0, "data": [x, y - y0]} for x, y in st.get('neg', [])]
    m = np.array(remove(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), session=new_session('sam'),
                        only_mask=True, sam_prompt=prompt))      # NB: kwarg is sam_prompt; input_points is silently ignored
    full = np.zeros(src.shape[:2], np.uint8); full[y0:y1] = m
    os.makedirs(os.path.join(work, 'sam'), exist_ok=True)
    cv2.imwrite(os.path.join(work, 'sam', '%s_%d.png' % (name, idx)), full)


def raw_mask(scene, spec, d, work, o, src):
    h, w = src.shape[:2]; m = np.zeros((h, w), bool); meth = o.get('method', 'sam')
    if meth == 'sam':
        for i in range(len(o['strips'])):
            subprocess.check_call([sys.executable, os.path.abspath(__file__), scene, '--sam-worker', o['name'], str(i)])
            m |= cv2.imread(os.path.join(work, 'sam', '%s_%d.png' % (o['name'], i)), 0) > 127
    elif meth == 'grabcut':
        gm = np.full((h, w), cv2.GC_BGD, np.uint8); x0, y0, x1, y1 = o['box']
        gm[y0:y1, x0:x1] = cv2.GC_PR_BGD
        gm[y0 + (y1 - y0) // 6:y1 - (y1 - y0) // 6, x0 + (x1 - x0) // 6:x1 - (x1 - x0) // 6] = cv2.GC_PR_FGD
        for x, y in o.get('fg', []): cv2.circle(gm, (x, y), 6, cv2.GC_FGD, -1)
        for x, y in o.get('bg', []): cv2.circle(gm, (x, y), 6, cv2.GC_BGD, -1)
        bg_, fg_ = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(src, gm, None, bg_, fg_, o.get('iters', 5), cv2.GC_INIT_WITH_MASK)
        m = (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)
    elif meth == 'color':
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array(o['hsv_lo']), np.array(o['hsv_hi'])) > 0
        if 'poly' in o:
            pm = np.zeros((h, w), np.uint8); cv2.fillPoly(pm, [np.array(o['poly'], np.int32)], 1); m &= pm > 0
    elif meth == 'polygon':
        mm = np.zeros((h, w), np.uint8); cv2.fillPoly(mm, [np.array(o['poly'], np.int32)], 1); m = mm > 0
    elif meth == 'rembg':
        from rembg import remove, new_session
        from PIL import Image
        r = np.array(remove(Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)), session=new_session(o.get('model', 'isnet-general-use')),
                            only_mask=True)); m = r > 127
    elif meth == 'file':
        m = cv2.resize(cv2.imread(os.path.join(d, o['path']), 0), (w, h)) > 127
    return m


def clean(o, m, src):
    h, w = m.shape; m = m.copy()
    mm = m.astype(np.uint8)
    for pts in o.get('add_poly', []): cv2.fillPoly(mm, [np.array(pts, np.int32)], 1)   # add_poly: list of polygons
    m = mm > 0
    for x0, y0, x1, y1 in o.get('cut_rects', []): m[y0:y1, x0:x1] = False
    m = m.astype(np.uint8)
    k = o.get('close', 7)
    if k: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(k))
    fh = o.get('fill_holes', 6000)                                   # enclosed holes smaller than this are jewellery/stripes, fill them
    if fh:
        n, lab, st, _ = cv2.connectedComponentsWithStats((1 - m).astype(np.uint8), connectivity=4)
        edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        for i in range(1, n):
            if i not in edge and st[i, 4] < fh: m[lab == i] = 1
    sp = o.get('specks', 300)
    if sp:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if st[i, 4] < sp: m[lab == i] = 0
    gc = o.get('global_close', 0)                                    # thin see-through gaps (bracelets, hair gaps): close big, keep only small new blobs
    if gc:
        mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(gc)); gap = (mc > 0) & (m == 0)
        n, lab, st, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=8)
        xr = o.get('gap_x', [0, w])
        for i in range(1, n):
            if st[i, 4] <= o.get('gap_max', 4000) and st[i, 0] >= xr[0] and st[i, 0] + st[i, 2] <= xr[1]: m[lab == i] = 1
    for x0, y0, x1, y1, kk in o.get('local_close', []):              # stronger closing only inside a box
        m[y0:y1, x0:x1] = cv2.morphologyEx(m[y0:y1, x0:x1], cv2.MORPH_CLOSE, ell(kk))
    g = m.astype(np.float32)
    if o.get('guided', True):
        g = cv2.ximgproc.guidedFilter(src, g, o.get('guided_r', 4), 1e-3)
        c = o.get('contrast', 2.2); g = np.clip((g - 0.5) * c + 0.5, 0, 1)
    return g


def build_frame(spec, d, work, src):
    f = spec.get('frame')
    if not f: return
    h, w = src.shape[:2]; FR = np.zeros((h, w), np.uint8)
    l, t, r, b = f.get('border', [0, 0, 0, 0])
    FR[:, :l] = 1; FR[:t] = 1
    if r: FR[:, w - r:] = 1
    if b: FR[h - b:] = 1
    gd = f.get('gold')
    if gd:                                                           # ornaments: colour-classify inside windows only
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        col = cv2.inRange(hsv, np.array(gd['hsv_lo']), np.array(gd['hsv_hi'])) > 0
        win = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in gd['windows']: win[y0:y1, x0:x1] = 1
        m = cv2.morphologyEx((col & win).astype(np.uint8), cv2.MORPH_CLOSE, ell(7))
        n, lab, st, _ = cv2.connectedComponentsWithStats(np.maximum(FR, m), connectivity=8)
        keep = np.zeros_like(FR)
        for i in range(1, n):
            c = lab == i
            if (c & (FR > 0)).any(): keep[c] = 1
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats((1 - keep).astype(np.uint8), connectivity=4)
        edge = set(np.unique(np.r_[lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
        for i in range(1, n2):
            if i not in edge and st2[i, 4] < gd.get('fill_holes', 900): keep[lab2 == i] = 1
        FR = keep
    for x, y, rr in f.get('circles', []): cv2.circle(FR, (x, y), rr, 1, -1)
    for x0, y0, x1, y1 in f.get('rects', []): FR[y0:y1, x0:x1] = 1
    np.save(os.path.join(work, 'frame.npy'), cv2.GaussianBlur(FR.astype(np.float32), (0, 0), 0.8))
    ov = src.copy(); ov[FR == 0] = (ov[FR == 0] * 0.35).astype(np.uint8)
    cv2.imwrite(os.path.join(work, 'frame_overlay.png'), ov); print('frame mask ->', work + '/frame.npy (+frame_overlay.png)')


def build_bg(spec, d, work, src, mattes, size):
    W, H = size; up = cv2.resize(src, (W, H), interpolation=cv2.INTER_LANCZOS4)
    lift = np.zeros((H, W), np.float32)
    for o in spec['objects']:
        if o.get('lift', True) and o['name'] in mattes: lift = np.maximum(lift, mattes[o['name']])
    md = cv2.dilate((lift > 0.3).astype(np.uint8), ell(9))
    half = cv2.resize(up, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    inp = cv2.inpaint(half, cv2.resize(md, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST), 6, cv2.INPAINT_TELEA)
    inp = cv2.resize(inp, (W, H), interpolation=cv2.INTER_CUBIC)
    a = cv2.GaussianBlur(md.astype(np.float32), (0, 0), 1.2)[..., None]
    cv2.imwrite(os.path.join(work, 'bg.png'), (up * (1 - a) + inp * a).astype(np.uint8)); print('bg plate ->', work + '/bg.png')


def main():
    if '--sam-worker' in sys.argv:
        sam_worker(sys.argv[1], sys.argv[3], int(sys.argv[4])); return
    scene = os.path.abspath(sys.argv[1]); spec, d, work = load(scene)
    only = [a for a in sys.argv[2:] if not a.startswith('--')]
    os.makedirs(os.path.join(work, 'mattes'), exist_ok=True)
    src = cv2.imread(os.path.join(d, spec['image'])); W, H = spec.get('out_size', [1080, 1920])
    mattes = {}
    for o in spec['objects']:
        p = os.path.join(work, 'mattes', o['name'] + '.npy')
        if (only and o['name'] not in only) or '--bg' in sys.argv:
            if os.path.exists(p): mattes[o['name']] = np.load(p)
            continue
        if o.get('method') == 'file' and o.get('soft'):          # ready-made soft matte (.png or .npy, any size): used as is
            pth = os.path.join(d, o['path'])
            m = np.load(pth) if pth.endswith('.npy') else cv2.imread(pth, 0).astype(np.float32) / 255
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA if m.shape[1] > W else cv2.INTER_CUBIC)
            np.save(p, m); mattes[o['name']] = m; print('matte', o['name'], '(soft file)'); continue
        m = clean(o, raw_mask(scene, spec, d, work, o, src), src)
        big = cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC); np.save(p, big); mattes[o['name']] = big
        ov = src.copy(); ov[..., 2] = np.clip(ov[..., 2] * (1 - 0.55 * m) + 255 * 0.55 * m, 0, 255)
        cv2.imwrite(os.path.join(work, 'mattes', o['name'] + '_overlay.png'), ov)
        print('matte', o['name'], 'coverage %.1f%%' % (100 * (m > 0.5).mean()))
    build_bg(spec, d, work, src, mattes, (W, H)); build_frame(spec, d, work, src)


if __name__ == '__main__':
    main()
