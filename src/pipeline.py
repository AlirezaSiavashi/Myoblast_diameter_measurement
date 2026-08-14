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
from skimage.filters import gaussian
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk, remove_small_holes
from skimage.measure import label as sklabel, regionprops

from model import load_model, prep_mhc, norm

Image.MAX_IMAGE_PIXELS = None
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PXUM_960 = 1.32          # px per um for 960x720 (downscaled) images
PXUM_1920 = 2.64         # px per um for 1920x1440 (native) images  (E52 Klara)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = os.path.join(PROJ, "output", "models", "emb_net.pt")

_K8 = np.ones((3, 3), int); _K8[1, 1] = 0


# ----------------------------------------------------------------------------- segmentation
def segment(net, mhc, thr=0.45, mhc_gate=True):
    """MHC image -> solid fiber mask. mhc_gate rejects nucleus-dense regions with no MHC support."""
    with torch.no_grad():
        sem, emb, ori = net(torch.from_numpy(prep_mhc(mhc))[None, None].float().to(DEVICE))
    prob = torch.sigmoid(sem)[0, 0].cpu().numpy()
    mask = binary_closing(remove_small_holes(remove_small_objects(prob > thr, 300), 2000), disk(4))
    if mhc_gate:
        mhc_s = gaussian(mhc.astype(np.float32), 1.5)
        mask = remove_small_objects(mask & (mhc_s > mhc_s.max() * 0.06), 300)
    return mask


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


def fibers(mask, pxum=PXUM_960, min_peel=45, nms=70, single=False):
    """
    Decompose the mask into per-fiber centerlines and measure each fiber's thickest diameter.

    single=True  -> one main spine per connected component (clean, may under-count touching fibers).
    single=False -> iterative longest-path 'peeling' + spatial NMS (recovers separate fibers; tune min_peel/nms).

    Returns list of dicts: {diameter_um, point(y,x), perp(dy,dx), path(Nx2)}.
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
        S = set(zip(ys.tolist(), xs.tolist()))
        while True:
            path = _longest_path(S)
            if path is None or len(path) < min_peel:
                break
            pw = [2 * dist[y, x] / pxum for (y, x) in path]     # local width along the spine
            i = int(np.argmax(pw)); y, x = path[i]; w = pw[i]   # thickest point (medial-axis caliber)
            if 3 <= w <= 140:
                seg = np.array(path[max(0, i - 6):i + 7], float)
                if len(seg) >= 3:
                    vt = np.linalg.svd(seg - seg.mean(0), full_matrices=False)[2]
                    perp = np.array([-vt[0][1], vt[0][0]])       # perpendicular to local spine tangent
                else:
                    perp = np.array([0.0, 1.0])
                out.append({"diameter_um": w, "point": (y, x), "perp": perp, "path": np.array(path)})
            for p in path:
                S.discard(p)
            if single:
                break
    if not single and nms:
        out = _nms(out, nms)
    return out


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


# ----------------------------------------------------------------------------- convenience
def load_field(folder):
    mhc = np.array(Image.open(os.path.join(folder, "Image_CH3.tif")).convert("RGB"))[:, :, 0]
    dapi = np.array(Image.open(os.path.join(folder, "Image_CH1.tif")).convert("RGB"))[:, :, 2].astype(np.float32)
    pxum = PXUM_1920 if mhc.shape[1] >= 1900 else PXUM_960
    return mhc, dapi, pxum


def run_field(folder, model=DEFAULT_MODEL, min_peel=45, nms=70, single=False, draw=True):
    """Full pipeline on one field folder. Returns dict of results; optionally saves an overlay PNG."""
    net = load_model(model, inp=1, device=DEVICE)
    mhc, dapi, pxum = load_field(folder)
    mask = segment(net, mhc)
    fibs = fibers(mask, pxum=pxum, min_peel=min_peel, nms=nms, single=single)
    diams = [f["diameter_um"] for f in fibs]
    fi, nin, ntot = fusion_index(mask, dapi)
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
            y, x = f["point"]; w = f["diameter_um"] * pxum; perp = f["perp"]
            for yy, xx in f["path"]:
                d.point((int(xx), int(yy)), fill=(0, 220, 255))
            dxp, dyp = perp[1] * w / 2, perp[0] * w / 2
            d.line([(x - dxp, y - dyp), (x + dxp, y + dyp)], fill=(255, 255, 0), width=2)
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
