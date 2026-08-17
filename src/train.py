"""
Train the weakly-supervised myotube segmentation model (MHNet, MHC-only).

Supervision comes entirely from the free instance masks in output/training_set/*.npz
(generated from the experts' caliper marks via SAM -- see build_training_set.py).

Losses:
  PU-masked BCE   : unlabeled/uncertain pixels are IGNORED, so unmeasured fibers are
                    never taught as background (naive training collapses without this).
  region Dice     : solid fills (no porous "Swiss-cheese" masks).
  cbDice          : centerline-boundary Dice -> width-aware topology (fixes the porosity/thinning
                    that plain clDice produces; this is the key ablation).
  discriminative  : De Brabandere instance-embedding loss (pull same fiber together, push apart).
  orientation MSE : predict local fiber direction (used to separate crossing fibers).

Run:  C:/mlenv/Scripts/python.exe src/train.py           (holds out E40 + E44 for testing)
Output: output/models/emb_net.pt
"""
import os, glob, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.feature import structure_tensor

from model import MHNet, prep_mhc, EMB

Image.MAX_IMAGE_PIXELS = None
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(PROJ, "output", "training_set")
MODELS = os.path.join(PROJ, "output", "models"); os.makedirs(MODELS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOLDOUT = ("E40", "E44")
NITER = int(os.environ.get("NITER", "6000"))
NFIELDS = int(os.environ.get("NFIELDS", "350"))


# --- soft-cbDice ---
def _erode(x): return -F.max_pool2d(-x, 3, 1, 1)
def _dilate(x): return F.max_pool2d(x, 3, 1, 1)
def _open(x): return _dilate(_erode(x))
def _skel(x, it=8):
    x1 = _open(x); s = F.relu(x - x1)
    for _ in range(it):
        x = _erode(x); x1 = _open(x); d = F.relu(x - x1); s = s + F.relu(d - s * d)
    return s
def dice(p, t, m): p = p * m; t = t * m; return 1 - (2 * (p * t).sum() + 1) / ((p * p).sum() + (t * t).sum() + 1)
def cbdice(p, t, w, m):
    p = p * m; t = t * m; sp = _skel(p); st = _skel(t)
    tp = (sp * t * w).sum() / ((sp * w).sum() + 1e-6); ts = (st * p * w).sum() / ((st * w).sum() + 1e-6)
    return 1 - 2 * tp * ts / (tp + ts + 1e-6)
def disc(emb, inst, dv=0.5, dd=1.5):
    tot = emb.sum() * 0; nb = 0
    for b in range(emb.shape[0]):
        e = emb[b]; lab = inst[b, 0].long(); ids = torch.unique(lab); ids = ids[ids > 0]
        if len(ids) < 1: continue
        means = []; lv = 0.
        for i in ids:
            m = (lab == i)
            if m.sum() < 5: continue
            ei = e[:, m]; mu = ei.mean(1); means.append(mu)
            lv = lv + F.relu(torch.norm(ei - mu[:, None], dim=0) - dv).pow(2).mean()
        if len(means) < 1: continue
        means = torch.stack(means, 0); lreg = torch.norm(means, dim=1).mean()
        ld = torch.tensor(0., device=DEVICE)
        if len(means) > 1:
            dm = torch.cdist(means, means); K = len(means); msk = ~torch.eye(K, dtype=bool, device=DEVICE)
            ld = F.relu(2 * dd - dm[msk]).pow(2).mean()
        tot = tot + lv / len(means) + ld + 0.001 * lreg; nb += 1
    return tot / max(nb, 1)
def orient(img):
    Arr, Arc, Acc = structure_tensor(img.astype(np.float32), sigma=3, order="rc")
    th = 0.5 * np.arctan2(2 * Arc, (Arr - Acc) + 1e-6)
    return np.stack([np.cos(2 * th), np.sin(2 * th)]).astype(np.float32)


def main():
    files = [f for f in glob.glob(TS + "/*.npz") if not any(h in os.path.basename(f) for h in HOLDOUT)]
    random.seed(2); random.shuffle(files)
    data = []
    for f in files:
        d = np.load(f, allow_pickle=True); ip = str(d["img_path"])
        if not os.path.exists(ip): continue
        g = np.array(Image.open(ip).convert("RGB"))[:, :, 0]
        if g.shape[1] >= 1900: continue
        pre = prep_mhc(g); fg = (d["labels"] > 0).astype(np.float32)
        valid = np.clip(fg + (pre < 0.06).astype(np.float32), 0, 1)         # PU mask: fg + confident bg
        rad = ndimage.distance_transform_edt(fg).astype(np.float32)         # cbDice width weight
        data.append((pre, fg, valid, rad, d["labels"].astype(np.int32)))
        if len(data) >= NFIELDS: break
    print(f"loaded {len(data)} fields (holdout {HOLDOUT})", flush=True)

    net = MHNet(40, inp=1).to(DEVICE); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, NITER)

    def crop(a, sz=256):
        pre, fg, valid, rad, inst = a; H, W = pre.shape
        y = random.randint(0, H - sz); x = random.randint(0, W - sz); sl = (slice(y, y + sz), slice(x, x + sz))
        return pre[sl].copy(), fg[sl].copy(), valid[sl].copy(), rad[sl].copy(), inst[sl].copy(), orient(pre[sl])

    def batch(bs=6):
        P = []; Fg = []; V = []; R = []; I = []; O = []
        for _ in range(bs):
            p, f, v, r, i, o = crop(random.choice(data))
            P.append(p); Fg.append(f); V.append(v); R.append(r); I.append(i); O.append(o)
        t = lambda a: torch.from_numpy(np.stack(a))[:, None].float().to(DEVICE)
        return t(P), t(Fg), t(V), t(R), t(I), torch.from_numpy(np.stack(O)).float().to(DEVICE)

    net.train(); t0 = time.time()
    for it in range(NITER):
        P, Fg, V, R, I, O = batch(6); sem, emb, ori, _ = net(P); prob = torch.sigmoid(sem)
        bce = (F.binary_cross_entropy_with_logits(sem, Fg, reduction="none") * V).sum() / (V.sum() + 1e-6)
        loss = (bce + 1.5 * dice(prob, Fg, V) + 0.6 * cbdice(prob, Fg, R, V)
                + 1.0 * disc(emb, I)
                + 0.3 * (((F.normalize(ori, dim=1) - O) ** 2).sum(1, keepdim=True) * Fg).sum() / (Fg.sum() + 1e-6))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0:
            print(f"  it{it}: loss={loss.item():.3f} [{time.time()-t0:.0f}s]", flush=True)
    torch.save(net.state_dict(), os.path.join(MODELS, "emb_net.pt"))
    print(f"DONE {(time.time()-t0)/60:.1f} min -> output/models/emb_net.pt", flush=True)


if __name__ == "__main__":
    main()
