"""
Held-out evaluation: field-mean diameter agreement with the expert (Katja / Klara).

Compares our per-field median diameter against the expert's Result*.txt values on the
held-out experiments (E40, E44), reporting Pearson r, ratio, and within-5um agreement.

Run:  C:/mlenv/Scripts/python.exe src/evaluate.py
"""
import os, re, glob
import numpy as np
from PIL import Image
from scipy.stats import pearsonr

from pipeline import load_model, segment, fibers, load_field, DEFAULT_MODEL, PROJ, PXUM_960

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.join(PROJ, "Katja Myoblasts")
HELDOUT = ["E40 hpMb L", "E44 hpMb G Klara"]


def parse_um(fp):
    v = []
    for enc in ("utf-8", "latin-1"):
        try:
            for ln in open(fp, encoding=enc):
                p = ln.split("\t")
                if len(p) >= 3 and p[0].strip().isdigit():
                    try:
                        x = float(p[2].strip())
                        if 0 < x < 400: v.append(x)
                    except ValueError:
                        pass
            return v
        except Exception:
            continue
    return v


def natkey(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def fields_under(base):
    out = []
    for dp, dn, fn in os.walk(base):
        res = sorted([f for f in fn if re.match(r"result", f, re.I) and f.lower().endswith(".txt")], key=natkey)
        subs = sorted([d for d in dn if os.path.exists(os.path.join(dp, d, "Image_CH3.tif"))
                       and "20x" in d.lower() and "4x" not in d.lower() and "opa" not in d.lower()], key=natkey)
        for rf, sd in zip(res, subs):
            out.append((os.path.join(dp, rf), os.path.join(dp, sd)))
    return out


def main(model=DEFAULT_MODEL, min_peel=45, nms=70, single=False, width_mode="disk"):
    net = load_model(model, inp=1)
    print(f"[width_mode={width_mode}]")
    for exp in HELDOUT:
        ours = []; katja = []
        for rf, fld in fields_under(os.path.join(ROOT, exp)):
            gt = parse_um(rf)
            if len(gt) < 3:
                continue
            mhc, dapi, pxum = load_field(fld)
            if mhc.shape[1] >= 1900:
                continue
            mask = segment(net, mhc)
            fibs = fibers(mask, pxum=pxum, min_peel=min_peel, nms=nms, single=single, width_mode=width_mode)
            diams = [f["diameter_um"] for f in fibs]
            if len(diams) < 3:
                continue
            ours.append(np.mean(diams)); katja.append(np.mean(gt))
        ours = np.array(ours); katja = np.array(katja)
        r = pearsonr(ours, katja)[0]
        print(f"{exp:18s}  fields={len(ours):3d}  r={r:.3f}  ratio={np.mean(ours/katja):.2f}  "
              f"within5um={100*np.mean(np.abs(ours-katja)<=5):.0f}%")
    print("\n(baseline Cellpose-SAM for reference: E40 r~0.70, E44 r~0.33)")


if __name__ == "__main__":
    main()
