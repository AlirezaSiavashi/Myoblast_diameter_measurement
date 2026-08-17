"""
Train the 2-channel (MHC + DAPI) model with the extra nucleus-membership head (mem_net.pt).

Same weak supervision as train.py (free instance masks from caliper marks) PLUS a membership head
supervised PU-style, only on nucleus pixels:
    nucleus (DAPI) INTERSECT fiber          -> in  (target 1)
    nucleus (DAPI) INTERSECT low-MHC bg     -> out (target 0)
    nucleus over MHC but not in a measured fiber -> IGNORED (uncertain; bootstrap-bounded)
So the head learns in/out (fusion index), not fiber-vs-background again.

Two things we test downstream (evaluate_mem.py):
  1. fusion index straight from the head (end-to-end), vs the Cellpose-in-mask method.
  2. whether the auxiliary DAPI + membership signal improves held-out E44 segmentation (diameter r).

Run:  C:/mlenv/Scripts/python.exe src/train_mem.py        (holds out E40 + E44)
Output: output/models/mem_net.pt
"""
import os, glob, time, random
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.morphology import remove_small_objects

from model import MHNet, prep_mhc, norm, EMB
from train import dice, cbdice, disc, orient          # reuse the exact loss helpers

Image.MAX_IMAGE_PIXELS = None
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(PROJ, "output", "training_set")
MODELS = os.path.join(PROJ, "output", "models"); os.makedirs(MODELS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOLDOUT = ("E40", "E44")
NITER = int(os.environ.get("NITER", "6000"))
NFIELDS = int(os.environ.get("NFIELDS", "350"))
NUC_THR = 0.30           # DAPI (normed) threshold for nucleus pixels


def load_dapi(mhc_path):
    dp = mhc_path.replace("Image_CH3", "Image_CH1")
    if not os.path.exists(dp):
        return None
    return np.array(Image.open(dp).convert("RGB"))[:, :, 2].astype(np.float32)


def main():
    files = [f for f in glob.glob(TS + "/*.npz") if not any(h in os.path.basename(f) for h in HOLDOUT)]
    random.seed(2); random.shuffle(files)
    data = []
    for f in files:
        d = np.load(f, allow_pickle=True); ip = str(d["img_path"])
        if not os.path.exists(ip):
            continue
        g = np.array(Image.open(ip).convert("RGB"))[:, :, 0]
        if g.shape[1] >= 1900:
            continue
        dapi = load_dapi(ip)
        if dapi is None:
            continue
        pre = prep_mhc(g); dn = norm(dapi)
        two = np.stack([pre, dn]).astype(np.float32)                        # (2,H,W) model input
        fg = (d["labels"] > 0).astype(np.float32)
        valid = np.clip(fg + (pre < 0.06).astype(np.float32), 0, 1)         # PU mask for sem loss
        rad = ndimage.distance_transform_edt(fg).astype(np.float32)         # cbDice width weight
        nuc = remove_small_objects(dn > NUC_THR, 20)                        # nucleus pixels (bootstrap)
        conf = (fg > 0) | (pre < 0.06)                                      # confident in OR confident out
        memmask = (nuc & conf).astype(np.float32)                          # PU-membership loss mask
        data.append((two, fg, valid, rad, d["labels"].astype(np.int32), memmask))
        if len(data) >= NFIELDS:
            break
    print(f"loaded {len(data)} fields (holdout {HOLDOUT}); mem-supervised px/field ~"
          f"{int(np.mean([m[5].sum() for m in data]))}", flush=True)

    net = MHNet(40, inp=2).to(DEVICE); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, NITER)

    def crop(a, sz=256):
        two, fg, valid, rad, inst, memmask = a; H, W = fg.shape
        y = random.randint(0, H - sz); x = random.randint(0, W - sz)
        s2 = (slice(None), slice(y, y + sz), slice(x, x + sz)); sl = (slice(y, y + sz), slice(x, x + sz))
        return (two[s2].copy(), fg[sl].copy(), valid[sl].copy(), rad[sl].copy(),
                inst[sl].copy(), memmask[sl].copy(), orient(two[0][sl]))

    def batch(bs=6):
        TW = []; Fg = []; V = []; R = []; I = []; M = []; O = []
        for _ in range(bs):
            tw, f, v, r, i, m, o = crop(random.choice(data))
            TW.append(tw); Fg.append(f); V.append(v); R.append(r); I.append(i); M.append(m); O.append(o)
        t = lambda a: torch.from_numpy(np.stack(a))[:, None].float().to(DEVICE)
        P = torch.from_numpy(np.stack(TW)).float().to(DEVICE)               # (B,2,sz,sz)
        return P, t(Fg), t(V), t(R), t(I), t(M), torch.from_numpy(np.stack(O)).float().to(DEVICE)

    net.train(); t0 = time.time()
    for it in range(NITER):
        P, Fg, V, R, I, M, O = batch(6); sem, emb, ori, mem = net(P); prob = torch.sigmoid(sem)
        bce = (F.binary_cross_entropy_with_logits(sem, Fg, reduction="none") * V).sum() / (V.sum() + 1e-6)
        membce = (F.binary_cross_entropy_with_logits(mem, Fg, reduction="none") * M).sum() / (M.sum() + 1e-6)
        loss = (bce + 1.5 * dice(prob, Fg, V) + 0.6 * cbdice(prob, Fg, R, V)
                + 1.0 * disc(emb, I)
                + 0.3 * (((F.normalize(ori, dim=1) - O) ** 2).sum(1, keepdim=True) * Fg).sum() / (Fg.sum() + 1e-6)
                + 1.0 * membce)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0:
            print(f"  it{it}: loss={loss.item():.3f} mem={membce.item():.3f} [{time.time()-t0:.0f}s]", flush=True)
    torch.save(net.state_dict(), os.path.join(MODELS, "mem_net.pt"))
    print(f"DONE {(time.time()-t0)/60:.1f} min -> output/models/mem_net.pt", flush=True)


if __name__ == "__main__":
    main()
