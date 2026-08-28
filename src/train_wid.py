"""
Train MHNetDS + LEARNED WIDTH HEAD (mem_net_wid.pt).

Why a learned width head. Diagnosis on the held-out outlines showed diameter error is dominated by
INSTANCE errors, not by the caliper geometry: matched-instance IoU is only 0.40-0.59, so small tubes
merged into a big instance inherit its width (+114%) and big tubes that get split are measured on a
sub-part (-15%). Deriving diameter from the skeleton of a possibly-wrong instance is therefore
brittle. Instead we regress LOCAL FIBER WIDTH PER PIXEL, supervised directly by the expert outlines
(target = 2 x distance-transform at the nearest polygon-skeleton point). A tube's diameter is then
read from the width map over its pixels, which stays correct even when the instance is split or
merged -- it decouples diameter from instance segmentation.

Three changes vs train_mem/consistency fine-tunes, all aimed at the LONG-FIBER problem
(the expert-reported failure: big myotubes split into several at random positions):

  1. ARCHITECTURE - gap-tolerant Dynamic Snake blocks + dilated bottleneck (see dsconv.py).
     The plain U-Net's receptive field (~140 px) is far smaller than a myotube (500-900 px), so it
     literally cannot see a whole fiber; the embedding drifts along the fiber and DBSCAN then cuts
     it. Snake blocks give ANISOTROPIC long-range context that follows the fiber, with directional
     momentum so it survives MHC signal dropouts.
  2. LARGER CROPS (512 vs 288) - so a whole large myotube fits in one training sample and the
     "one fiber = one embedding" constraint is actually enforced at fiber scale.
  3. STRONG AUGMENTATION incl. FIBER-GAP DROPOUT - random erasures placed ON fibers, which is the
     supervision signal that teaches the snake to bridge gaps (and mimics weak-staining regions).

Warm-started from mem_net_cons2.pt; snake/dilated layers are zero-init (exact identity) so training
starts from the current best model and only improves on it.

Run:  C:/mlenv/Scripts/python.exe src/train_wid.py
Out:  output/models/mem_net_wid.pt
"""
import struct, os, sys, zipfile, glob, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage

from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

from model import load_model, prep2, prep_mhc, norm
from train import disc, dice

Image.MAX_IMAGE_PIXELS = None
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(PROJ, "Katja Myoblasts")
OUTLINE = os.path.join(BASE, "myotube outline")
MODELS = os.path.join(PROJ, "output", "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAP = {
    1: r"E47 hpMb L_L2\L2\3d\IQ3+Wnt7a\20x_04", 2: r"E52 hpMb G\3d\scrambled\20x_5",
    3: r"E52 hpMb G\3d\scrambled\20x_3", 4: r"E52 hpMb G\3d\IQ2\20x_3", 5: r"E52 hpMb G\5d\IQ3\20x_04",
    6: r"E52 hpMb G\5d\IQ3\20x_02", 7: r"E47 hpMb L_L2\L2\5d\scrambled\20x_01",
    8: r"E47 hpMb L_L2\L2\3d\IQ2+Rapamycin\20x_02", 9: r"E47 hpMb L_L2\L2\3d\IQ2+Rapamycin\20x_01",
    10: r"E47 hpMb L_L2\L2\3d\IQ3_Rapamycin\20x_04", 11: r"E47 hpMb L_L2\L2\3d\IQ3_Rapamycin\20x_01",
    12: r"E47 hpMb L_L2\L2\5d\IQ3+Wnt7a\20x_04", 13: r"E47 hpMb L_L2\L2\5d\IQ3+Wnt7a\20x_08",
    14: r"E47 hpMb L_L2\L2\3d\DMSO\20x_04", 15: r"E47 hpMb L_L2\L2\5d\DMSO\20x_03",
    16: r"E47 hpMb L_L2\L2\5d\Wnt7a\20x_01", 17: r"E47 hpMb L_L2\L2\5d\Wnt7a\20x_03",
    18: r"E47 hpMb L_L2\L2\5d\scrambled\20x_05",
    19: r"E44 hpMb G Klara\5d\IQGAP2\20x_1", 20: r"E44 hpMb G Klara\5d\IQGAP 3\20x_5",
    21: r"E44 hpMb G Klara\3d\IGQAP 3\20x_23", 22: r"E44 hpMb G Klara\3d\SCR siRNA\20x_24",
    23: r"E40 hpMb L\scrambled 5d\20x_05", 24: r"E40 hpMb L\IQ2 5d\20x_04",
    25: r"E40 hpMb L\IQ2 3d\20x_04", 26: r"E40 hpMb L\scrambled 3d\20x_05",
    27: r"E47 hpMb L_L2\L2\3d\IQ3+Wnt7a\20x_06", 28: r"E47 hpMb L_L2\L2\3d\IQ3+Wnt7a\20x_03",
    29: r"E47 hpMb L_L2\L2\3d\IQ3+Wnt7a\20x_01", 30: r"E44 hpMb G Klara\5d\IQGAP 3\20x_2",
    31: r"E40 hpMb L\IQ3 3d\20x_01", 32: r"E40 hpMb L\IQ3 3d\20x_03",
    33: r"E40 hpMb L\IQ3 3d\20x_05", 34: r"E40 hpMb L\IQ3 3d\20x_02",
}
HELD = {22, 24, 25, 30, 34}
NITER = int(os.environ.get("NITER", "2000"))
SZ = int(os.environ.get("SZ", "512"))


def parse_roi(b):
    top, left, bottom, right = struct.unpack(">hhhh", b[8:16]); n = struct.unpack(">h", b[16:18])[0]
    if n < 3: return None
    base = 64
    xs = [struct.unpack(">h", b[base + 2 * i:base + 2 * i + 2])[0] for i in range(n)]
    ys = [struct.unpack(">h", b[base + 2 * n + 2 * i:base + 2 * n + 2 * i + 2])[0] for i in range(n)]
    return [(left + x, top + y) for x, y in zip(xs, ys)]


def load_polys(folder):
    P = []
    z = os.path.join(folder, "RoiSet.zip")
    if os.path.exists(z):
        zf = zipfile.ZipFile(z)
        for nm in zf.namelist():
            p = parse_roi(zf.read(nm))
            if p: P.append(p)
    for rf in glob.glob(os.path.join(folder, "*.roi")):
        p = parse_roi(open(rf, "rb").read())
        if p: P.append(p)
    return P


PXUM = 1.32   # px per um for the 960x720 fields
WSCALE = 50.0  # width head regresses width/WSCALE so targets are O(1) (raw um failed to converge)


def width_map(inst):
    """Per-pixel local fiber width (um) from the expert polygons: 2 x EDT at the nearest
    polygon-skeleton point, so the whole cross-section carries the tube's local width."""
    W = np.zeros(inst.shape, np.float32)
    for k in range(1, int(inst.max()) + 1):
        m = inst == k
        if m.sum() < 20:
            continue
        edt = ndimage.distance_transform_edt(m)
        sk = skeletonize(m)
        if not sk.any():
            continue
        sy, sx = np.nonzero(sk)
        py, px = np.nonzero(m)
        _, idx = cKDTree(np.c_[sy, sx]).query(np.c_[py, px])
        W[py, px] = 2.0 * edt[sy[idx], sx[idx]] / PXUM / WSCALE
    return W


def build():
    data = []
    for pic in [k for k in MAP if k not in HELD]:
        fld = os.path.join(BASE, MAP[pic])
        polys = load_polys(os.path.join(OUTLINE, f"picture {pic}"))
        if not polys: continue
        mhc = np.array(Image.open(os.path.join(fld, "Image_CH3.tif")).convert("RGB"))[:, :, 0]
        dapi = np.array(Image.open(os.path.join(fld, "Image_CH1.tif")).convert("RGB"))[:, :, 2].astype(np.float32)
        inst = np.zeros((720, 960), np.int32)
        for k, p in enumerate(polys, 1):
            m = Image.new("L", (960, 720), 0); ImageDraw.Draw(m).polygon(p, fill=1)
            inst[np.array(m, bool)] = k
        data.append((mhc, dapi, inst, width_map(inst)))
    return data


def augment(mhc, dapi, inst, wid, rng):
    """Geometric + photometric augmentation, plus FIBER-GAP DROPOUT (erase patches ON fibers)."""
    if rng.random() < 0.5: mhc, dapi, inst, wid = mhc[:, ::-1], dapi[:, ::-1], inst[:, ::-1], wid[:, ::-1]
    if rng.random() < 0.5: mhc, dapi, inst, wid = mhc[::-1], dapi[::-1], inst[::-1], wid[::-1]
    if rng.random() < 0.5:
        mhc, dapi, inst, wid = np.rot90(mhc), np.rot90(dapi), np.rot90(inst), np.rot90(wid)
    mhc = mhc.astype(np.float32).copy(); dapi = dapi.astype(np.float32).copy()
    inst = inst.copy(); wid = wid.copy()
    # photometric on MHC: gain, gamma, noise (labels unchanged -> teaches intensity invariance)
    mhc *= rng.uniform(0.7, 1.4)
    mhc = np.clip(mhc, 0, 255) ** rng.uniform(0.8, 1.25)
    mhc += rng.normal(0, rng.uniform(0, 4), mhc.shape)
    dapi *= rng.uniform(0.8, 1.25)
    # FIBER-GAP DROPOUT: dim random blobs sitting on fibers, labels untouched
    fib = inst > 0
    if fib.any() and rng.random() < 0.8:
        ys, xs = np.nonzero(fib)
        for _ in range(rng.integers(1, 6)):
            j = rng.integers(len(ys)); cy, cx = ys[j], xs[j]
            ry, rx = rng.integers(4, 16), rng.integers(4, 16)
            y0, y1 = max(0, cy - ry), min(mhc.shape[0], cy + ry)
            x0, x1 = max(0, cx - rx), min(mhc.shape[1], cx + rx)
            mhc[y0:y1, x0:x1] *= rng.uniform(0.05, 0.45)          # signal dropout, fiber still labelled
    return np.clip(mhc, 0, 255), np.clip(dapi, 0, None), inst, wid


def main():
    data = build()
    print(f"loaded {len(data)} TRAIN fields (held out {sorted(HELD)}), crop {SZ}", flush=True)
    net = load_model(os.path.join(MODELS, "mem_net_ds.pt"), inp=2, arch="ds", device=DEVICE)
    net.train()
    mw = float(np.mean([d[3][d[3] > 0].mean() for d in data]))       # mean normalized width
    torch.nn.init.zeros_(net.wid.weight); torch.nn.init.constant_(net.wid.bias, mw)
    print(f"width head initialised at mean width {mw*WSCALE:.1f} um", flush=True)
    head = list(net.wid.parameters()); hid = {id(p) for p in head}
    opt = torch.optim.Adam([{"params": [p for p in net.parameters() if id(p) not in hid], "lr": 1e-4},
                            {"params": head, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, NITER)
    rng = np.random.default_rng(0)

    def sample():
        mhc, dapi, inst, wid = random.choice(data)
        mhc, dapi, inst, wid = augment(mhc, dapi, inst, wid, rng)
        H, W = inst.shape
        sz = min(SZ, H, W)
        y = rng.integers(0, H - sz + 1); x = rng.integers(0, W - sz + 1)
        sl = (slice(y, y + sz), slice(x, x + sz))
        m, d, i, w = mhc[sl], dapi[sl], inst[sl], wid[sl]
        pre = prep_mhc(m)
        two = np.stack([pre, norm(d)]).astype(np.float32)
        fg = (i > 0).astype(np.float32)
        valid = np.clip(fg + (pre < 0.06).astype(np.float32), 0, 1)
        return two, fg, valid, i.astype(np.int32), w.astype(np.float32)

    def batch(bs=3):
        P = []; Fg = []; V = []; I = []; Wd = []
        for _ in range(bs):
            a, b, c, d, e = sample(); P.append(a); Fg.append(b); V.append(c); I.append(d); Wd.append(e)
        t = lambda z: torch.from_numpy(np.stack(z))[:, None].float().to(DEVICE)
        return torch.from_numpy(np.stack(P)).float().to(DEVICE), t(Fg), t(V), t(I), t(Wd)

    t0 = time.time()
    for it in range(NITER):
        P, Fg, V, I, Wd = batch()
        sem, emb, ori, mem, wpred = net(P); prob = torch.sigmoid(sem)
        bce = (F.binary_cross_entropy_with_logits(sem, Fg, reduction="none") * V).sum() / (V.sum() + 1e-6)
        # width regression: smooth-L1 on fiber pixels only (um units), scaled to ~O(1)
        wloss = (F.smooth_l1_loss(wpred, Wd, reduction="none", beta=0.1) * Fg).sum() / (Fg.sum() + 1e-6)
        loss = bce + 1.0 * dice(prob, Fg, V) + 2.5 * disc(emb, I, dv=0.3, dd=1.5) + 3.0 * wloss
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 250 == 0:
            print(f"  it{it}: loss={loss.item():.3f} wid_um={wloss.item()*50:.1f} [{time.time()-t0:.0f}s]", flush=True)
    torch.save(net.state_dict(), os.path.join(MODELS, "mem_net_wid.pt"))
    print(f"DONE {(time.time()-t0)/60:.1f} min -> output/models/mem_net_wid.pt", flush=True)


if __name__ == "__main__":
    main()
