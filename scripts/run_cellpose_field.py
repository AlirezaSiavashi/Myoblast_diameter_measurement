"""
Run Cellpose-SAM nucleus detection on the DAPI channel of ONE field and save:
  - <name>_DAPI_cellpose_overlay.png   (outlines on DAPI, for visual QC)
  - <name>_DAPI_cellpose_masks.tif     (integer label mask -> feeds regionprops)
  - <name>_COMPARISON_cellpose_vs_markers.png  (if a MYOG+DAPI marker exists)
  - <name>_result.json                 (counts + agreement with manual count)

Usage (from the C:\\mlenv venv):
  C:/mlenv/Scripts/python.exe run_cellpose_field.py <field_folder> <field_basename> [manual_DAPI_count]

Example:
  C:/mlenv/Scripts/python.exe run_cellpose_field.py \
    "C:/.../Erik RMS/250807_IF_.../250807_RD_n2_KIF5_20x_04" 250807_RD_n2_KIF5_20x_04 841
"""
import sys, os, json, time, glob
import numpy as np
from PIL import Image, ImageDraw
import tifffile
from cellpose import models, utils
Image.MAX_IMAGE_PIXELS = None

fld  = sys.argv[1]
name = sys.argv[2]
gt   = int(sys.argv[3]) if len(sys.argv) > 3 else None
out  = r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output"
os.makedirs(out, exist_ok=True)

# --- locate the DAPI channel file (exclude marker files) ---
dapi_files = [f for f in glob.glob(os.path.join(fld, "*DAPI*.tif"))
              if "marker" not in os.path.basename(f).lower()
              and "+" not in os.path.basename(f)]
dapi_path = dapi_files[0]
dapi_rgb = np.array(Image.open(dapi_path).convert("RGB"))
dapi = dapi_rgb[:, :, 2].astype(np.float32)          # DAPI signal = blue channel
print("DAPI:", os.path.basename(dapi_path), dapi.shape)

# --- run Cellpose-SAM (default cpsam model, CPU) ---
t = time.time()
model = models.CellposeModel(gpu=False)
masks, flows, styles = model.eval(dapi, batch_size=8,
                                  flow_threshold=0.4, cellprob_threshold=0.0)
n = int(masks.max())
dt = time.time() - t
print(f"Cellpose-SAM count: {n} nuclei  ({dt:.1f}s)")

tifffile.imwrite(os.path.join(out, f"{name}_DAPI_cellpose_masks.tif"), masks.astype(np.uint16))

# --- overlay outlines on DAPI ---
g = dapi - dapi.min(); g = (g / (g.max() + 1e-6) * 255).astype(np.uint8)
rgb = np.stack([g] * 3, -1).copy()
ov = Image.fromarray(rgb); d = ImageDraw.Draw(ov)
for o in utils.outlines_list(masks):
    if len(o) > 1:
        d.line([tuple(p) for p in o] + [tuple(o[0])], fill=(255, 255, 0), width=1)
d.text((10, 10), f"Cellpose-SAM: {n} nuclei", fill=(0, 255, 0))
ov.save(os.path.join(out, f"{name}_DAPI_cellpose_overlay.png"))

# --- side-by-side vs Erik's MYOG+DAPI marker (all nuclei = red + yellow dots), if present ---
mk = glob.glob(os.path.join(fld, "*MYOG*DAPI*marker*.tif")) + \
     glob.glob(os.path.join(fld, "*MYOG*marker*.tif"))
if mk:
    mkimg = Image.open(mk[0]).convert("RGB")
    comp = Image.new("RGB", (g.shape[1] * 2 + 20, g.shape[0] + 30), (0, 0, 0))
    comp.paste(ov, (0, 30)); comp.paste(mkimg, (g.shape[1] + 20, 30))
    d2 = ImageDraw.Draw(comp)
    d2.text((10, 8), f"Cellpose-SAM: {n}", fill=(0, 255, 0))
    d2.text((g.shape[1] + 30, 8), "Erik markers (red=MYOG+, yellow=MYOG-)", fill=(255, 255, 255))
    comp.save(os.path.join(out, f"{name}_COMPARISON_cellpose_vs_markers.png"))

res = {"field": name, "model": "Cellpose-SAM (cpsam v4, CPU)",
       "cellpose_count": n, "runtime_sec": round(dt, 1)}
if gt is not None:
    res.update({"manual_DAPI_count": gt, "difference": n - gt,
                "percent_of_manual": round(100 * n / gt, 1),
                "abs_percent_error": round(100 * abs(n - gt) / gt, 1)})
json.dump(res, open(os.path.join(out, f"{name}_result.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
