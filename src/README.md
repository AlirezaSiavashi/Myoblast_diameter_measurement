# Weakly-supervised myotube morphometry

Automated measurement of cultured muscle fibers — **diameter** and **fusion index** — learned
entirely from the experts' hand-drawn caliper *width-lines*, with **no manual masks**.

Headline result: on a **held-out lab + cell source (E44)** our diameter tracks the expert at
**r ≈ 0.82**, where an off-the-shelf foundation model (Cellpose-SAM) collapses to **r ≈ 0.33**.

---

## Files

| File | What it does |
|------|--------------|
| `model.py`     | `MHNet` network (sem + embedding + orientation heads) and preprocessing. |
| `pipeline.py`  | **Inference**: segment → per-fiber centerlines → diameter (caliber) + fusion index. Runnable on one field. |
| `train.py`     | Train `MHNet` from the free instance masks (cbDice + region + PU + embedding + orientation). |
| `evaluate.py`  | Held-out diameter agreement vs the expert on E40 + E44 (Pearson r, ratio, within-5 µm). |

## Environments (persist on disk)

| Env | For |
|-----|-----|
| `C:\mlenv`     | Cellpose-SAM (py3.13, CUDA) — used by everything here. |
| `C:\omni311`   | Omnipose baseline. |
| `C:\myosam311` | MyoSAM baseline (needs the checkpoint — not downloaded). |

Run anything with e.g. `C:/mlenv/Scripts/python.exe src/pipeline.py <field-folder>`.

## Models (`output/models/`)

| Checkpoint | Input | Notes |
|------------|-------|-------|
| **`emb_net.pt`** | MHC only | **Default.** No nucleus false-positives (a fiber needs MHC support). |
| `mc_net.pt`      | MHC+DAPI+Myogenin | Better E44 calibration, but can hallucinate fibers in nucleus-dense regions. |
| `stage2b_unet.pt`| MHC | Earlier semantic-only model (cbDice). |

## How the pipeline works

1. **Segment** (`segment`) — `MHNet` predicts a fiber probability map; threshold + close +
   an **MHC-gate** (a fiber must have red signal → kills nucleus-only false positives).
2. **Centerlines** (`fibers`) — skeletonize, **prune spurs** (removes the little end-fans that
   create false junctions), then extract per-fiber spines:
   - `single=False` (**peel**, default): iterative **longest-path** extraction + spatial **NMS**.
     Measures every separate fiber → best diameter accuracy.
   - `single=True`: one main spine per connected component → cleanest one-per-fiber visual,
     matches the expert's *count/median*, but noisier field-mean (fewer measurements).
3. **Diameter** — medial-axis **caliber**: thickest point on the spine (2×distance-transform),
   drawn **perpendicular** to the local tangent. Bounded, so it can't run away at merges.
4. **Fusion index** (`fusion_index`) — Cellpose on DAPI → fraction of nuclei inside the fiber mask.

## Results (held-out E40 + E44)

| Method | E40 r | E44 r |
|--------|-------|-------|
| Cellpose-SAM (off-the-shelf) | 0.70 | **0.33** |
| **Ours (peel)** | **0.80** | **0.82** |

- **Diameter distribution** matches the expert (day-3 median ~24 vs 24 µm).
- **Edge localization** (HD-style, at the expert's caliper endpoints): median ~2–3 µm.
- **Fusion index** automated (~33% on day-3 controls).
- **Ablation**: clDice → porous masks (r 0.38 raw); **cbDice** → solid (r 0.84 raw).

## Honest limitations

- **Per-fiber precision is label-limited.** The expert gave width *values* but **no coordinates
  and no fiber outlines**, so per-fiber segmentation width cannot be trained/validated against her,
  and HD95 vs her is only computable at the sparse caliper points.
- **Count vs distribution trade-off.** `peel` gives the best diameter *r* but over-counts vs the
  expert's *selection*; `single` matches her count/median but has a noisier field-mean.
- **Day-5 fused sheets** can merge many fibers into one instance (a large-diameter outlier).
- The single thing that would raise per-fiber accuracy and enable proper boundary metrics is a
  small set of **expert fiber outlines** (~15–20 fields).

## Reproduce

```bash
# one field (prints diameters + fusion index, saves an overlay)
C:/mlenv/Scripts/python.exe src/pipeline.py "Katja Myoblasts/E40 hpMb L/scrambled 3d/20x_01"

# held-out diameter agreement (r, ratio, within-5um)
C:/mlenv/Scripts/python.exe src/evaluate.py

# retrain the model (holds out E40 + E44; ~15 min on GPU)
C:/mlenv/Scripts/python.exe src/train.py
```

Training data (`output/training_set/*.npz`, 12,836 free masks) is generated from the caliper marks
via SAM; the generator script lives in the session scratchpad and should be moved here too if needed.
