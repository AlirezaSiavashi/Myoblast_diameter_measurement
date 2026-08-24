"""
Myotube morphometry — full inference pipeline.

    image (MHC + DAPI channels)
      -> segment()      : semantic fiber mask (model + MHC-gate)
      -> fibers()       : per-fiber centerlines  (prune spurs -> longest-path peeling -> spatial NMS)
      -> diameter()     : thickest perpendicular width per fiber (medial-axis caliber)
      -> fusion_index() : nuclei inside fibers / total nuclei (needs Cellpose on DAPI)

Run on one field:
    C:/mlenv/Scripts/python.exe src/pipeline.py "Katja Myoblasts/E40 hpMb L/scrambled 3d/20x_01"

Notes / current status (be honest with yourself):
  * Diameter DISTRIBUTION matches the expert well (median ~24 vs 24 um on day-3 fields).
  * Diameter GENERALISES across lab/source: held-out E44 r=0.83 vs off-the-shelf Cellpose-SAM r=0.33.
  * Fiber COUNT is approximate and tunable (MIN_PEEL / NMS). Exact match to the expert's
    belly+arms decomposition needs her fiber OUTLINES, which do not exist yet.
  * Day-5 fused sheets can merge many fibers into one large instance (a known hard case).
"""
import os, sys, glob
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.ndimage import convolve, binary_dilation
from scipy.spatial import cKDTree
from skimage.filters import gaussian
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk, remove_small_holes
from skimage.measure import label as sklabel, regionprops

from model import load_model, prep_mhc, prep2, norm

Image.MAX_IMAGE_PIXELS = None
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PXUM_960 = 1.32          # px per um for 960x720 (downscaled) images
PXUM_1920 = 2.64         # px per um for 1920x1440 (native) images  (E52 Klara)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = os.path.join(PROJ, "output", "models", "emb_net.pt")   # MHC-only (inp=1)
MEM_MODEL = os.path.join(PROJ, "output", "models", "mem_net.pt")       # MHC+DAPI + membership head (inp=2)
MEM_CONS_MODEL = os.path.join(PROJ, "output", "models", "mem_net_cons.pt")  # embedding fine-tuned for long-fiber
# consistency (tighter intra-fiber margin) -> fragments big myotubes less; run_field(model=MEM_CONS_MODEL)

_K8 = np.ones((3, 3), int); _K8[1, 1] = 0


# ----------------------------------------------------------------------------- segmentation
def segment(net, mhc, thr=0.45, mhc_gate=True):
    """MHC image -> solid fiber mask. mhc_gate rejects nucleus-dense regions with no MHC support."""
    with torch.no_grad():
        sem, emb, ori, mem = net(torch.from_numpy(prep_mhc(mhc))[None, None].float().to(DEVICE))
    prob = torch.sigmoid(sem)[0, 0].cpu().numpy()
    mask = binary_closing(remove_small_holes(remove_small_objects(prob > thr, 300), 2000), disk(4))
    if mhc_gate:
        mhc_s = gaussian(mhc.astype(np.float32), 1.5)
        mask = remove_small_objects(mask & (mhc_s > mhc_s.max() * 0.06), 300)
    return mask


def _refine_with_membership(mask, memp, thr=0.6):
    """Grow the mask into confident in-fiber pixels (memp>thr) that touch it -> recovers faint
    fiber bridges the semantic head missed; then re-fill holes / drop specks."""
    grow = binary_dilation(mask, iterations=2) & (memp > thr)
    m = remove_small_holes(mask | grow, 2000)
    return remove_small_objects(m, 300)


def segment2(net, mhc, dapi, thr=0.45, mhc_gate=True, refine=False, return_feats=False):
    """2-channel (MHC+DAPI) model -> (fiber mask, per-pixel in-fiber membership prob).
    refine=True uses the membership head to grow the mask into in-fiber nuclei (faint-fiber recovery).
    return_feats=True also returns (emb (8,H,W), ori (2,H,W)) for embedding-based instance separation."""
    with torch.no_grad():
        sem, emb, ori, mem = net(torch.from_numpy(prep2(mhc, dapi))[None].float().to(DEVICE))
    prob = torch.sigmoid(sem)[0, 0].cpu().numpy()
    memp = torch.sigmoid(mem)[0, 0].cpu().numpy()
    mask = binary_closing(remove_small_holes(remove_small_objects(prob > thr, 300), 2000), disk(4))
    if mhc_gate:
        mhc_s = gaussian(mhc.astype(np.float32), 1.5)
        mask = remove_small_objects(mask & (mhc_s > mhc_s.max() * 0.06), 300)
    if refine:
        mask = _refine_with_membership(mask, memp)
    if return_feats:
        return mask, memp, emb[0].cpu().numpy(), ori[0].cpu().numpy()
    return mask, memp


def prune_spurs(skel, min_len=9):
    """Remove short dead-end skeleton branches (spurs) that create false junctions."""
    sk = skel.copy()
    for _ in range(25):
        nb = convolve(sk.astype(int), _K8, mode="constant")
        endpoints = sk & (nb == 1)
        branch = binary_dilation(sk & (nb >= 3), iterations=1)
        seg = sklabel(sk & ~branch, connectivity=2)
        removed = False
        for r in regionprops(seg):
            ys, xs = r.coords[:, 0], r.coords[:, 1]
            if endpoints[ys, xs].any() and r.area < min_len:
                sk[ys, xs] = False; removed = True
        if not removed:
            break
    return sk

    
def _neigh(p, S):
    y, x = p
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and (y + dy, x + dx) in S:
                yield (y + dy, x + dx)


def _bfs_far(s, S):
    dd = {s: 0}; par = {s: None}; q = deque([s])
    while q:
        u = q.popleft()
        for v in _neigh(u, S):
            if v not in dd:
                dd[v] = dd[u] + 1; par[v] = u; q.append(v)
    return max(dd, key=dd.get), par


def _longest_path(S):
    """Longest geodesic path across all connected components of pixel-set S."""
    if not S:
        return None
    seen = set(); best = None
    for seed in list(S):
        if seed in seen:
            continue
        comp = {seed}; seen.add(seed); q = deque([seed])
        while q:
            u = q.popleft()
            for v in _neigh(u, S):
                if v not in comp:
                    comp.add(v); seen.add(v); q.append(v)
        A, _ = _bfs_far(seed, comp)
        B, par = _bfs_far(A, comp)
        path = []; cur = B
        while cur is not None:
            path.append(cur); cur = par[cur]
        if best is None or len(path) > len(best):
            best = path
    return best


def _cast(mask, y, x, dy, dx, maxr):
    """March from (y,x) along (dy,dx) in 0.5-px steps; return distance where we leave the mask."""
    H, W = mask.shape
    r = 0.0
    while r < maxr:
        yy = int(round(y + dy * r)); xx = int(round(x + dx * r))
        if yy < 0 or yy >= H or xx < 0 or xx >= W or not mask[yy, xx]:
            return r
        r += 0.5
    return r


def _edge_width(mask, y, x, perp, dist, cap=2.5, ksym=1.8, sweep=(-0.20, -0.10, 0.0, 0.10, 0.20)):
    """True perpendicular edge-to-edge width (px) at (y,x): cast a ray BOTH ways along the
    spine-perpendicular until each side leaves the mask, so an off-centre spine still gets the
    real width (near-half + far-half) instead of 2*nearest-edge. A few small angle offsets are
    tried and the SHORTEST span kept (local thickness, robust to tangent-estimation error).
    Each side is capped at cap*dist+3 px, and the far half is clamped to ksym*near so the ray
    can't blow through into a crossing fiber at a merge (a real tube is only mildly asymmetric)."""
    maxr = cap * dist[y, x] + 3.0
    dy0, dx0 = float(perp[0]), float(perp[1])
    best = None
    for a in sweep:
        ca, sa = np.cos(a), np.sin(a)
        dy = dy0 * ca - dx0 * sa
        dx = dy0 * sa + dx0 * ca
        s1 = _cast(mask, y, x, dy, dx, maxr); s2 = _cast(mask, y, x, -dy, -dx, maxr)
        if best is None or s1 + s2 < best[0] + best[1]:
            best = (s1, s2)
    near, far = min(best), max(best)
    return near + min(far, ksym * near)


def _cast_pt(mask, y, x, dy, dx, maxr):
    """March from (y,x) along (dy,dx); return (distance, last-inside (y,x)) at the mask edge."""
    H, W = mask.shape
    r = 0.0; ly, lx = float(y), float(x)
    while r < maxr:
        yy = int(round(y + dy * r)); xx = int(round(x + dx * r))
        if yy < 0 or yy >= H or xx < 0 or xx >= W or not mask[yy, xx]:
            return r, (ly, lx)
        ly, lx = y + dy * r, x + dx * r; r += 0.5
    return r, (ly, lx)


def _perp_endpoints(y, x, perp, w_px):
    """Caliper endpoints for the non-chord modes: w_px long, centred at (y,x), along perp."""
    hy, hx = perp[0] * w_px / 2.0, perp[1] * w_px / 2.0
    return (y - hy, x - hx), (y + hy, x + hx)


def _perp_chord(mask, y, x, perp, dist, cap=2.6, sweep=(-0.15, -0.075, 0.0, 0.075, 0.15)):
    """Edge-to-edge width ACROSS the fiber: cast along the spine-perpendicular (with a small
    angular refine, keeping the shortest span so it stays perpendicular, not along the fiber) and
    return (width_px, e1(y,x), e2(y,x)) -- endpoints are the two mask-boundary hits, so the caliper
    is drawn flush to the mask. Unlike a global minimal chord this does NOT under-measure irregular
    boundaries, because it only searches a narrow cone around the true perpendicular."""
    maxr = cap * dist[y, x] + 3.0
    dy0, dx0 = float(perp[0]), float(perp[1])
    best = None
    for a in sweep:
        ca, sa = np.cos(a), np.sin(a)
        dy = dy0 * ca - dx0 * sa; dx = dy0 * sa + dx0 * ca
        r1, e1 = _cast_pt(mask, y, x, dy, dx, maxr); r2, e2 = _cast_pt(mask, y, x, -dy, -dx, maxr)
        if best is None or r1 + r2 < best[0]:
            best = (r1 + r2, e1, e2)
    return best


def fibers(mask, pxum=PXUM_960, min_peel=45, nms=70, single=False, width_mode="chord",
           jfac=1.0, emar=4, nuclei=None, min_nuc=0):
    """
    Decompose the mask into per-fiber centerlines and measure each fiber's thickest diameter.

    single=True  -> one main spine per connected component (clean, may under-count touching fibers).
    single=False -> iterative longest-path 'peeling' + spatial NMS (recovers separate fibers; tune min_peel/nms).

    Where we measure: on the spine, but with JUNCTIONS/BELLIES EXCLUDED (vessel-segment style) --
    a spine point is a candidate only if its distance to the nearest branch point is >= jfac*radius
    (so the caliper lands on a real fiber arm, not the fused blob), and not within emar of a spine end.
    We take the thickest such point.

    width_mode="chord" (default) -> minimal edge-to-edge chord = true local width; caliper endpoints
                                    are the two mask-boundary hits, so the drawn line is flush to the mask.
    width_mode="edge"            -> perpendicular edge-to-edge chord along the spine-perpendicular.
    width_mode="disk"            -> 2*distance-transform (symmetric assumption; legacy).

    nuclei / min_nuc : OPTIONAL myotube-validity gate. nuclei is an (N,2) array of nucleus centroids
        (y,x); a fiber is kept only if >= min_nuc of them fall within its body-band. Default OFF
        (min_nuc=0). A myotube is multinucleated, so this removes nucleus-free spurious segments and
        improves COUNT realism -- but held-out tests show it does NOT improve diameter agreement (it
        also drops real thin arms that share nuclei with the belly, biasing the mean up), so it is left
        off for diameter. Useful for count / fusion. (Myogenin-only nuclei gave no gain over DAPI.)

    Returns list of dicts: {diameter_um, point(y,x), perp(dy,dx), p1(y,x), p2(y,x), path(Nx2)}.
    """
    dist = ndimage.distance_transform_edt(mask)
    lab = sklabel(mask)
    out = []
    for rr in regionprops(lab):
        cc = lab == rr.label
        sk = prune_spurs(skeletonize(cc), 9)
        ys, xs = np.where(sk)
        if len(xs) < min_peel:
            continue
        nb = convolve(sk.astype(int), _K8, mode="constant")
        bp = sk & (nb >= 3)                                      # skeleton branch points (junctions)
        d_branch = ndimage.distance_transform_edt(~bp) if bp.any() else None
        S = set(zip(ys.tolist(), xs.tolist()))
        while True:
            path = _longest_path(S)
            if path is None or len(path) < min_peel:
                break
            fib = _measure_path(path, mask, dist, d_branch, pxum, width_mode, jfac, emar)
            if fib is not None:
                out.append(fib)
            for p in path:
                S.discard(p)
            if single:
                break
    if min_nuc and nuclei is not None and len(nuclei):
        cent = np.asarray(nuclei, float)
        out = [f for f in out if _nuc_in_fiber(f["path"], dist, cent) >= min_nuc]
    if not single and nms:
        out = _nms(out, nms)
    return out


def _measure_path(path, mask, dist, d_branch, pxum, width_mode="chord", jfac=1.0, emar=4):
    """Measure one fiber spine `path` (ordered list of (y,x)): pick the thickest point on a clean
    arm (junctions/bellies excluded via d_branch), then the perpendicular edge-to-edge width there.
    Returns a fiber dict {diameter_um, point, perp, p1, p2, path} or None if width is out of range."""
    cand = [k for k in range(emar, len(path) - emar)
            if d_branch is None or d_branch[path[k]] >= jfac * dist[path[k]]]
    if not cand:
        cand = list(range(len(path)))
    i = max(cand, key=lambda k: dist[path[k]])                # thickest point on a clean arm
    y, x = path[i]; w = 2 * dist[y, x] / pxum
    seg = np.array(path[max(0, i - 6):i + 7], float)
    if len(seg) >= 3:
        vt = np.linalg.svd(seg - seg.mean(0), full_matrices=False)[2]
        perp = np.array([-vt[0][1], vt[0][0]])               # perpendicular to local spine tangent
    else:
        perp = np.array([0.0, 1.0])
    p1, p2 = _perp_endpoints(y, x, perp, w * pxum)
    if width_mode == "chord":
        span, e1, e2 = _perp_chord(mask, y, x, perp, dist)
        wc = span / pxum
        if 3 <= wc <= 140:
            w = wc; p1, p2 = e1, e2
    elif width_mode == "edge":
        we = _edge_width(mask, y, x, perp, dist) / pxum
        if 3 <= we <= 140:
            w = we; p1, p2 = _perp_endpoints(y, x, perp, w * pxum)
    if 3 <= w <= 140:
        return {"diameter_um": w, "point": (y, x), "perp": perp, "p1": p1, "p2": p2, "path": np.array(path)}
    return None


def fibers_embed(mask, emb, pxum=PXUM_960, S=60.0, w_emb=2.0, eps=1.3, min_samples=5,
                 minlen=25, width_mode="chord", jfac=1.0, emar=4):
    """
    Instance separation by DBSCAN clustering of skeleton pixels in a LEARNED feature space:
    [spatial (y,x)/S, w_emb * instance-embedding(8-D)]. The embedding head (trained with the
    discriminative loss to make same-fiber pixels similar) splits touching/fused fibers that the
    geometric skeleton merges. Each cluster is one myotube; its pixels' longest path is measured
    like fibers(). Validated on 196 expert outlines: count MAE 12.4->3.4, r 0.10->0.83, instance
    F1 0.21->0.41 vs the peel+NMS baseline. Orientation added nothing (embedding already separates).

    emb : (8, H, W) instance-embedding array from the network (see segment2(return_feats=True)).
    Returns the same list-of-dicts as fibers().
    """
    dist = ndimage.distance_transform_edt(mask)
    sk = prune_spurs(skeletonize(mask), 9)
    ys, xs = np.where(sk)
    if len(ys) < minlen:
        return []
    nb = convolve(sk.astype(int), _K8, mode="constant")
    bp = sk & (nb >= 3)
    d_branch = ndimage.distance_transform_edt(~bp) if bp.any() else None
    E = np.asarray(emb, np.float32)[:, ys, xs].T                      # (N, 8)
    feat = np.concatenate([np.c_[ys, xs].astype(np.float32) / S, E * w_emb], 1)
    from sklearn.cluster import DBSCAN
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feat)
    out = []
    for c in set(lab):
        if c == -1:
            continue
        m = lab == c
        if m.sum() < minlen:
            continue
        path = _longest_path(set(zip(ys[m].tolist(), xs[m].tolist())))
        if path is None or len(path) < minlen:
            continue
        fib = _measure_path(path, mask, dist, d_branch, pxum, width_mode, jfac, emar)
        if fib is not None:
            out.append(fib)
    return out


def _nuc_in_fiber(path, dist, cent, margin=4.0):
    """Number of nucleus centroids whose nearest spine point is within (local radius + margin)."""
    d, idx = cKDTree(path).query(cent)
    rad = dist[path[idx][:, 0], path[idx][:, 1]]
    return int((d <= rad + margin).sum())


def _nms(fibs, dist_thr):
    fibs = sorted(fibs, key=lambda f: -len(f["path"]))
    kept = []
    for f in fibs:
        y, x = f["point"]
        if all((y - ky) ** 2 + (x - kx) ** 2 > dist_thr ** 2 for (ky, kx) in [k["point"] for k in kept]):
            kept.append(f)
    return kept


# ----------------------------------------------------------------------------- fusion index
def fusion_index(mask, dapi, cp_model=None):
    """Fraction of DAPI nuclei whose centroid falls inside the fiber mask."""
    from cellpose import models
    if cp_model is None:
        cp_model = models.CellposeModel(gpu=(DEVICE == "cuda"))
    nuc, _, _ = cp_model.eval(dapi.astype(np.float32), flow_threshold=0.4, cellprob_threshold=0.0)
    n = int(nuc.max())
    if n == 0:
        return float("nan"), 0, 0
    cent = np.array(ndimage.center_of_mass(np.ones_like(nuc), nuc, range(1, n + 1))).astype(int)
    inside = int(mask[cent[:, 0], cent[:, 1]].sum())
    return 100.0 * inside / n, inside, n


def fusion_index_head(memp, dapi, cp_model=None, mem_thr=0.5):
    """End-to-end fusion index from the membership head: Cellpose finds nuclei on DAPI, and each
    nucleus is counted 'in' if its MEAN membership probability exceeds mem_thr (no fiber mask needed)."""
    from cellpose import models
    if cp_model is None:
        cp_model = models.CellposeModel(gpu=(DEVICE == "cuda"))
    nuc, _, _ = cp_model.eval(dapi.astype(np.float32), flow_threshold=0.4, cellprob_threshold=0.0)
    n = int(nuc.max())
    if n == 0:
        return float("nan"), 0, 0
    means = np.array(ndimage.mean(memp, nuc, range(1, n + 1)))
    inside = int(np.sum(means > mem_thr))
    return 100.0 * inside / n, inside, n


# ----------------------------------------------------------------------------- convenience
def load_field(folder):
    mhc = np.array(Image.open(os.path.join(folder, "Image_CH3.tif")).convert("RGB"))[:, :, 0]
    dapi = np.array(Image.open(os.path.join(folder, "Image_CH1.tif")).convert("RGB"))[:, :, 2].astype(np.float32)
    pxum = PXUM_1920 if mhc.shape[1] >= 1900 else PXUM_960
    return mhc, dapi, pxum


def run_field(folder, model=None, min_peel=45, nms=70, single=False, draw=True,
              membership=True, refine=True):
    """Full pipeline on one field folder. Returns dict of results; optionally saves an overlay PNG.

    membership=True (default, headline): 2-channel mem_net (MHC+DAPI) -> membership-refined mask
        + end-to-end fusion index from the membership head (best held-out lab generalization).
    membership=False: MHC-only emb_net -> mask + Cellpose-in-mask fusion index (the older path).
    """
    mhc, dapi, pxum = load_field(folder)
    if membership:
        net = load_model(model or MEM_MODEL, inp=2, device=DEVICE)
        mask, memp, emb, ori = segment2(net, mhc, dapi, refine=refine, return_feats=True)
        fi, nin, ntot = fusion_index_head(memp, dapi)
        fibs = fibers_embed(mask, emb, pxum=pxum)          # DBSCAN embedding instance separation (best count)
    else:
        net = load_model(model or DEFAULT_MODEL, inp=1, device=DEVICE)
        mask = segment(net, mhc)
        fi, nin, ntot = fusion_index(mask, dapi)
        fibs = fibers(mask, pxum=pxum, min_peel=min_peel, nms=nms, single=single)
    diams = [f["diameter_um"] for f in fibs]
    res = {"folder": folder, "n_fibers": len(fibs),
           "median_diameter_um": float(np.median(diams)) if diams else float("nan"),
           "mean_diameter_um": float(np.mean(diams)) if diams else float("nan"),
           "diameters_um": [round(float(d), 1) for d in sorted(diams, reverse=True)],
           "fusion_index_pct": round(fi, 1), "nuclei_in_fibers": nin, "nuclei_total": ntot}
    if draw:
        from PIL import ImageDraw
        disp = ((mhc - mhc.min()) / (mhc.max() - mhc.min() + 1e-6) * 255).astype(np.uint8)
        db = (np.clip(dapi / np.percentile(dapi, 99.5), 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(np.dstack([disp, np.zeros_like(disp), db])); d = ImageDraw.Draw(im)
        for f in fibs:
            y, x = f["point"]
            for yy, xx in f["path"]:
                d.point((int(xx), int(yy)), fill=(0, 220, 255))
            (y1, x1), (y2, x2) = f["p1"], f["p2"]                 # caliper flush to the mask edges
            d.line([(x1, y1), (x2, y2)], fill=(255, 255, 0), width=2)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))
            d.text((int(x) + 4, int(y) - 7), f"{f['diameter_um']:.0f}", fill=(150, 255, 150))
        out_dir = os.path.join(PROJ, "output", "pipeline_out"); os.makedirs(out_dir, exist_ok=True)
        res["overlay"] = os.path.join(out_dir, os.path.basename(folder) + "_result.png")
        im.save(res["overlay"])
    return res


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "Katja Myoblasts/E40 hpMb L/scrambled 3d/20x_01"
    if not os.path.isabs(folder):
        folder = os.path.join(PROJ, folder)
    r = run_field(folder)
    print(f"folder            : {os.path.relpath(r['folder'], PROJ)}")
    print(f"fibers            : {r['n_fibers']}")
    print(f"median diameter   : {r['median_diameter_um']:.1f} um")
    print(f"diameters (um)    : {r['diameters_um']}")
    print(f"fusion index      : {r['fusion_index_pct']:.1f}%  ({r['nuclei_in_fibers']}/{r['nuclei_total']} nuclei)")
    if "overlay" in r:
        print(f"overlay saved     : {os.path.relpath(r['overlay'], PROJ)}")
