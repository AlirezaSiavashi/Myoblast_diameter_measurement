"""
Qualitative comparison for Katja: OUR headline pipeline (mem_net + membership-refine) vs Katja's own
marked overlays, on held-out fields from E40 and the domain-shift lab E44 (r 0.88 there).

Left column  = ours: green fiber outline + yellow thickest-diameter calipers + fusion index (from head).
Right column = Katja's marked image (her calipers + nuclei marks) for the SAME field.

Run:  C:/mlenv/Scripts/python.exe src/viz_katja.py
Out:  output/for_katja/mem_vs_katja_fields.png
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from pipeline import (load_model, segment2, fibers, fusion_index_head, load_field,
                      MEM_MODEL, PROJ, norm)

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.join(PROJ, "Katja Myoblasts")
OUT = os.path.join(PROJ, "output", "for_katja"); os.makedirs(OUT, exist_ok=True)

PICKS = [
    ("E40 (near-training)", os.path.join(ROOT, "E40 hpMb L", "IQ2 3d", "20x_01")),
    ("E40 (near-training)", os.path.join(ROOT, "E40 hpMb L", "IQ3 3d", "20x_03")),
    ("E44 Klara (held-out lab)", os.path.join(ROOT, "E44 hpMb G Klara", "3d", "BSA", "20x_21")),
    ("E44 Klara (held-out lab)", os.path.join(ROOT, "E44 hpMb G Klara", "3d", "DMSO", "20x_22")),
]


def marked(fld):
    for pat in ("*Overlay m*.tif", "*Overlay*measure*.tif", "*measure*.tif"):
        g = glob.glob(os.path.join(fld, pat))
        if g:
            return g[0]
    return None


net = load_model(MEM_MODEL, inp=2)
from cellpose import models
cp = models.CellposeModel(gpu=True)


def our_panel(fld):
    mhc, dapi, pxum = load_field(fld)
    mask, memp = segment2(net, mhc, dapi, refine=True)
    fibs = fibers(mask, pxum=pxum)
    fi, nin, ntot = fusion_index_head(memp, dapi, cp)
    disp = ((mhc - mhc.min()) / (mhc.max() - mhc.min() + 1e-6) * 255).astype(np.uint8)
    db = (norm(dapi) * 255).astype(np.uint8)
    base = np.dstack([disp, np.zeros_like(disp), db])                 # red MHC + blue DAPI (like her overlay)
    edge = mask ^ ndimage.binary_erosion(mask, iterations=2)          # thick fiber outline
    base[edge] = [0, 230, 0]                                          # green = our segmentation boundary
    im = Image.fromarray(base); d = ImageDraw.Draw(im)
    for f in fibs:
        y, x = f["point"]
        (y1, x1), (y2, x2) = f["p1"], f["p2"]                    # caliper flush to the mask edges
        d.line([(x1, y1), (x2, y2)], fill=(255, 255, 0), width=2)
        d.text((int(x) + 3, int(y) - 7), f"{f['diameter_um']:.0f}", fill=(180, 255, 180))
    med = np.median([f["diameter_um"] for f in fibs]) if fibs else float("nan")
    d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
    d.text((4, 5), f"OURS: {len(fibs)} fibers   median {med:.0f} um   fusion index {fi:.0f}%", fill=(255, 255, 255))
    return im


W, H = 640, 480
rows = []
for tag, fld in PICKS:
    mk = marked(fld)
    if not os.path.exists(os.path.join(fld, "Image_CH3.tif")) or mk is None:
        print("skip (missing CH3 or marked):", fld); continue
    ours = our_panel(fld).resize((W, H))
    kj = Image.open(mk).convert("RGB").resize((W, H))
    dk = ImageDraw.Draw(kj); dk.rectangle([0, 0, W, 20], fill=(0, 0, 0))
    dk.text((4, 5), "KATJA: calipers + nuclei marks (ground truth)", fill=(255, 255, 255))
    rows.append((tag + "   |   " + os.path.relpath(fld, ROOT), ours, kj))
    print("rendered:", os.path.relpath(fld, ROOT))

gap = 26
canvas = Image.new("RGB", (W * 2 + 10, len(rows) * (H + gap) + 6), (15, 15, 15))
d = ImageDraw.Draw(canvas)
for i, (name, ours, kj) in enumerate(rows):
    y0 = i * (H + gap) + gap
    canvas.paste(ours, (0, y0)); canvas.paste(kj, (W + 10, y0))
    d.text((6, i * (H + gap) + 6), name, fill=(255, 255, 0))
path = os.path.join(OUT, "mem_vs_katja_fields.png")
canvas.save(path)
print("saved ->", os.path.relpath(path, PROJ))
