"""
Held-out evaluation of the 2-channel membership model (mem_net.pt) vs the MHC-only baseline.

Two questions:
  1. SEGMENTATION: does adding DAPI + the membership head change held-out diameter agreement
     (E40, E44) vs the MHC-only emb_net? (baseline: E40 r=0.80, E44 r=0.82)
  2. FUSION INDEX: does the end-to-end head-based fusion index agree with the Cellpose-in-mask
     method? (no expert GT for fusion index -> we report method agreement + distributions.)

Run:  C:/mlenv/Scripts/python.exe src/evaluate_mem.py
"""
import os
import numpy as np
from PIL import Image
from scipy.stats import pearsonr

from pipeline import (load_model, segment2, fibers, fusion_index, fusion_index_head,
                      load_field, PROJ)
from evaluate import fields_under, parse_um, HELDOUT, ROOT

Image.MAX_IMAGE_PIXELS = None
MEM_MODEL = os.path.join(PROJ, "output", "models", "mem_net.pt")
BASELINE = "MHC-only emb_net baseline: E40 r=0.80 ratio=1.09 | E44 r=0.82 ratio=0.84"


def diameter_eval(net, refine=False, width_mode="disk"):
    tag = f"refine={refine}"
    for exp in HELDOUT:
        ours = []; katja = []
        for rf, fld in fields_under(os.path.join(ROOT, exp)):
            gt = parse_um(rf)
            if len(gt) < 3:
                continue
            mhc, dapi, pxum = load_field(fld)
            if mhc.shape[1] >= 1900:
                continue
            mask, _ = segment2(net, mhc, dapi, refine=refine)
            diams = [f["diameter_um"] for f in fibers(mask, pxum=pxum, width_mode=width_mode)]
            if len(diams) < 3:
                continue
            ours.append(np.mean(diams)); katja.append(np.mean(gt))
        ours = np.array(ours); katja = np.array(katja); r = pearsonr(ours, katja)[0]
        print(f"  [{tag}] {exp:18s} fields={len(ours):3d}  r={r:.3f}  "
              f"ratio={np.mean(ours/katja):.2f}  within5um={100*np.mean(np.abs(ours-katja)<=5):.0f}%")


def fusion_eval(net, per_exp=15):
    from cellpose import models
    cp = models.CellposeModel(gpu=True)
    for exp in HELDOUT:
        head = []; msk = []
        for rf, fld in fields_under(os.path.join(ROOT, exp))[:per_exp]:
            mhc, dapi, pxum = load_field(fld)
            if mhc.shape[1] >= 1900:
                continue
            mask, memp = segment2(net, mhc, dapi)
            fh, _, _ = fusion_index_head(memp, dapi, cp)
            fm, _, _ = fusion_index(mask, dapi, cp)
            if np.isnan(fh) or np.isnan(fm):
                continue
            head.append(fh); msk.append(fm)
        head = np.array(head); msk = np.array(msk)
        r = pearsonr(head, msk)[0] if len(head) > 2 else float("nan")
        print(f"  {exp:18s} n={len(head):3d}  FI_head={np.mean(head):.1f}%  FI_mask={np.mean(msk):.1f}%  "
              f"|diff|={np.mean(np.abs(head-msk)):.1f}pp  r(head,mask)={r:.2f}")


def main():
    print(BASELINE)
    net = load_model(MEM_MODEL, inp=2)
    print("\n[mem_net] DIAMETER (does DAPI+membership help segmentation?)")
    diameter_eval(net, refine=False)
    diameter_eval(net, refine=True)
    print("\n[mem_net] FUSION INDEX (end-to-end head vs Cellpose-in-mask; no expert GT)")
    fusion_eval(net)


if __name__ == "__main__":
    main()
