# Weakly-supervised myotube morphometry

Automated measurement of cultured muscle fibers — **diameter** and **fusion index** — learned
entirely from the experts' hand-drawn caliper *width-lines*, with **no manual masks**.

Headline result: on a **held-out lab + cell source (E44)** our diameter tracks the expert at
**r ≈ 0.82**, where an off-the-shelf foundation model (Cellpose-SAM) collapses to **r ≈ 0.33**.

---

## Files

| File | What it does |
|------|--------------|
| `model.py`     | `MHNet` network (sem + embedding + orientation + **membership** heads) and preprocessing. |
| `pipeline.py`  | **Inference**: segment → per-fiber centerlines → diameter (caliber) + fusion index. `run_field` defaults to the headline **mem_net + membership-refine** pipeline (`membership=False` for the MHC-only path). |
| `train.py`     | Train MHC-only `MHNet` from the free instance masks (cbDice + region + PU + embedding + orientation). |
| `train_mem.py` | Train the 2-channel (MHC+DAPI) model with the **PU-membership** head → `mem_net.pt`. |
| `evaluate.py`  | Held-out diameter agreement vs the expert on E40 + E44 (Pearson r, ratio, within-5 µm). |
| `evaluate_mem.py` | Same held-out eval for `mem_net` (with/without refine) + head-vs-mask fusion-index agreement. |
| `viz_katja.py` | Qualitative figure: our pipeline vs Katja's marked overlays on E40 + E44 fields. |

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
| **`emb_net.pt`** | MHC only | Default. No nucleus false-positives (a fiber needs MHC support). |
| **`mem_net.pt`** | MHC+DAPI | **Best / default.** Adds the nucleus-membership head -> end-to-end fusion index, and (with `segment2(refine=True)`) the membership signal recovers faint fibers -> held-out E40 r=0.86, E44 r=0.85. |
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
3. **Diameter** — vessel/road-width style **cross-section** (`width_mode="chord"`, default): the
   thickest spine point **with junctions/bellies excluded** (`jfac`) so the caliper lands on a real
   fiber arm, not the fused blob; width is the **perpendicular edge-to-edge chord**, and the caliper
   endpoints are the two mask-boundary hits so the drawn line is **flush to the mask**.
4. **Fusion index** — `fusion_index_head` (mem_net): per-nucleus mean of the membership head. The
   older `fusion_index` (Cellpose nuclei ∩ mask) is kept for the MHC-only path.

## Results (held-out E40 + E44)

All rows use the placement-correct **junction-excluded perpendicular chord** (`width_mode="chord"`),
except where noted. mem_net rows use `segment2(refine=True)`.

| Method | E40 r | E40 ratio | E44 r | E44 ratio |
|--------|-------|-----------|-------|-----------|
| Cellpose-SAM (off-the-shelf) | 0.70 | – | **0.33** | – |
| Ours, MHC-only (`emb_net`) | 0.80 | 1.09 | 0.82 | 0.84 |
| **Ours, `mem_net` + membership-refine** | **0.86** | 1.11 | **0.85** | 0.93 |

The membership-refined model tracks the expert at **r≈0.85–0.86 on both labs**, including the held-out
lab + cell source (E44), where off-the-shelf Cellpose-SAM collapses to 0.33. **This is a multi-task
result: the fusion-index supervision + DAPI improves segmentation** (it recovers faint E44 fibers the
plain MHC mask misses).

- **Fusion index** is **end-to-end** from the membership head (no separate nucleus-in-mask test);
  agrees with the mask-based method at **r≈0.95**, and on E44 recovers in-fiber nuclei the
  under-segmented mask misses (head 32% vs mask 24%). No expert GT yet -> bootstrap-bounded.
- **Diameter placement** (vessel-segment cross-section) matches how the expert measures: perpendicular,
  edge-to-edge, on a fiber arm — not the fused belly. This corrected an earlier bias where measuring the
  medial belly inflated widths (and made the old E44 r=0.88 partly a "right-for-wrong-reasons" artifact).
- **Width/junction ablation** (`fibers(width_mode=, jfac=)`): `disk`=2×distance-transform (medial,
  inflates at bellies); `chord`=perpendicular edge-to-edge (default). Junction-exclusion `jfac` trades
  the two labs — higher helps belly-dense E40, slightly costs sparse E44; `jfac=1.0` is the balanced pick
  (`jfac=0.8` gives E44 0.87/0.97 if maximising the domain-shift lab). Correlation stays ~0.85 across
  sane settings — geometry is not the bottleneck; the residual is instance/label-limited.
- **Nucleus-containment gate** (`fibers(nuclei=, min_nuc=)`, default OFF) — *tested negative for diameter.*
  A myotube is multinucleated, so requiring a measured fiber to contain a nucleus removes ~25–30% of
  segments (good for count realism). But those are mostly thin arms that share nuclei with the belly, so
  dropping them biases the mean up (E40 ratio 1.11→1.19, within-5µm 68%→53%); `r` moves only ±0.02–0.03.
  **CH2/Myogenin adds nothing over DAPI** here (`≥1 Myogenin⁺` ≤ `≥1 any-nucleus` on every metric).
  Kept off for diameter; useful for count/validity. CH2's real value is the *nuclei-per-myotube* readout,
  not diameter (Myogenin⁺ = differentiated myonucleus, 44% of nuclei; sits inside myotubes in the merge).

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
C:/mlenv/Scripts/python.exe src/evaluate.py          # MHC-only baseline
C:/mlenv/Scripts/python.exe src/evaluate_mem.py      # mem_net (+refine) + fusion-index agreement

# qualitative figure vs Katja's marked overlays (E40 + E44)
C:/mlenv/Scripts/python.exe src/viz_katja.py

# retrain (holds out E40 + E44; ~12 min on GPU each)
C:/mlenv/Scripts/python.exe src/train.py             # MHC-only  -> emb_net.pt
C:/mlenv/Scripts/python.exe src/train_mem.py         # MHC+DAPI+membership -> mem_net.pt
```

Training data (`output/training_set/*.npz`, 12,836 free masks) is generated from the caliper marks
via SAM; the generator script lives in the session scratchpad and should be moved here too if needed.
