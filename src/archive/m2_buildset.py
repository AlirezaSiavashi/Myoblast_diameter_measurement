import numpy as np, os, re, glob, csv, time, warnings
warnings.filterwarnings("ignore")
from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy.stats import pearsonr
Image.MAX_IMAGE_PIXELS=None
ROOT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/Katja Myoblasts"
OUT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/training_set"
os.makedirs(OUT,exist_ok=True)
MAX=int(os.environ.get("MAX_FIELDS","0"))

def natkey(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)',s)]
def parse_um(fp):
    v=[]
    for enc in("utf-8","latin-1"):
        try:
            for ln in open(fp,encoding=enc):
                p=ln.split("\t")
                if len(p)>=3 and p[0].strip().isdigit():
                    try:
                        x=float(p[2].strip())
                        if 0<x<400: v.append(x)
                    except: pass
            return v
        except: continue
    return v
def caliper_centers(clean,mark):
    diff=np.abs(mark.astype(int)-clean.astype(int)).max(2)
    mk=ndimage.binary_closing(diff>40,iterations=1)
    cs=[]
    for r in regionprops(label(mk)):
        if r.area<40 or r.eccentricity<0.70: continue
        ys,xs=r.coords[:,0],r.coords[:,1]; col=mark[ys,xs].astype(int).mean(0)
        if col[1]>col[0]+25 and col[1]>col[2]+25: continue
        cy,cx=r.centroid
        c=r.coords.astype(float); c-=c.mean(0)
        try:
            _,_,vt=np.linalg.svd(c,full_matrices=False); L=float((c@vt[0]).ptp())
        except Exception: L=float(max(r.image.shape))
        cs.append((cx,cy,max(L-2,3)))
    return cs
def belly(m,pxum):
    ys,xs=np.where(m)
    if len(xs)<3: return 0.0
    sub=m[ys.min():ys.max()+1, xs.min():xs.max()+1]
    d=ndimage.distance_transform_edt(sub); sk=skeletonize(sub)
    return 2*(d[sk].max() if sk.sum()>=3 else d.max())/pxum

tasks=[]
for dp,dn,fn in os.walk(ROOT):
    res=[f for f in fn if re.match(r'result',f,re.I) and f.lower().endswith('.txt')]
    subs=[d for d in dn if os.path.exists(os.path.join(dp,d,"Image_Overlay.tif"))]
    if res and subs:
        res.sort(key=natkey); subs.sort(key=natkey)
        for rf,sd in zip(res,subs): tasks.append((dp,rf,sd))
if MAX: tasks=tasks[:MAX]

sam=sam_model_registry["vit_b"](checkpoint="C:/sam_ckpt/sam_vit_b.pth").to("cuda")
pred=SamPredictor(sam)
rows=[]; t0=time.time(); nfield=0; nmask_total=0
for dp,rf,sd in tasks:
    field=os.path.join(dp,sd)
    marked=glob.glob(field+"/*measure*.tif")+glob.glob(field+"/*Overlay m*.tif")+glob.glob(field+"/*Overlay  m*.tif")
    cp=field+"/Image_Overlay.tif"; ch3=field+"/Image_CH3.tif"
    if not marked or not os.path.exists(cp) or not os.path.exists(ch3): continue
    gt=sorted([w for w in parse_um(os.path.join(dp,rf)) if 0<w<400],reverse=True)
    if len(gt)<3: continue
    try:
        clean=np.array(Image.open(cp).convert("RGB")); mark=np.array(Image.open(marked[0]).convert("RGB"))
        mhc=np.array(Image.open(ch3).convert("RGB"))[:,:,0]
        if clean.shape!=mark.shape: continue
        pxum=2.64 if mhc.shape[1]>=1900 else 1.32
        centers=caliper_centers(clean,mark)
        if len(centers)<3: continue
        pred.set_image(np.dstack([mhc]*3))
        lab=np.zeros(mhc.shape,np.uint16); sam_w=[]
        for (x,y,Lpx) in sorted(centers,key=lambda c:-mhc[int(c[1]),int(c[0])]):
            Lt=Lpx/pxum
            masks,scores,_=pred.predict(point_coords=np.array([[x,y]]),point_labels=np.array([1]),multimask_output=True)
            best=None; bestd=1e9; bestw=0
            for mi in range(masks.shape[0]):
                m=masks[mi]
                if m.sum()<150 or m.mean()>0.4: continue
                w=belly(m,pxum)
                if w<3 or w>150: continue
                d=abs(w-Lt)
                if d<bestd: bestd=d; best=m; bestw=w
            if best is None or bestw>2.5*Lt or bestw<0.3*Lt: continue
            k=int(lab.max())+1; lab[best&(lab==0)]=k; sam_w.append(bestw)
        nmask=int(lab.max())
        if nmask<3: continue
        order=np.argsort(-np.array(sam_w)); we=np.full(nmask,np.nan); gg=gt[:nmask]
        for rank,idx in enumerate(order):
            if rank<len(gg): we[idx]=gg[rank]
        exp=os.path.relpath(field,ROOT).split(os.sep)[0]
        key=re.sub(r'[^A-Za-z0-9]+','_',os.path.relpath(field,ROOT))
        np.savez_compressed(os.path.join(OUT,key+".npz"),labels=lab,
            sam_width=np.array(sam_w,np.float32),katja_width=we.astype(np.float32),img_path=ch3,pxum=pxum)
        rows.append([exp,os.path.relpath(field,ROOT),pxum,len(centers),nmask,
                     round(float(np.mean(sam_w)),1),round(float(np.mean(gt)),1)])
        nfield+=1; nmask_total+=nmask
        if nfield%50==0: print(f"  {nfield} fields | {nmask_total} masks | {time.time()-t0:.0f}s",flush=True)
    except Exception: continue
with open(OUT+"/manifest.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["exp","field","pxum","n_caliper","n_mask","sam_mean_w","katja_mean_w"]); w.writerows(rows)
sw=np.array([r[5] for r in rows]); kw=np.array([r[6] for r in rows])
print(f"\n=== TRAINING SET BUILT ===")
print(f"fields: {nfield} | fiber masks: {nmask_total} | time {time.time()-t0:.0f}s")
if len(sw)>3: print(f"SAM-width vs Katja (per-field): r={pearsonr(sw,kw)[0]:.3f} | ratio={np.mean(sw/kw):.2f} | bias={np.mean(sw-kw):+.1f}um")
print(f"saved -> {OUT}")
