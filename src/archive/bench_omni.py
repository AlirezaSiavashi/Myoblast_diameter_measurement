import numpy as np, os, re, glob, csv, time, warnings
warnings.filterwarnings("ignore")
from PIL import Image
from scipy import ndimage
from skimage.filters import gaussian
from skimage.morphology import skeletonize
from skimage.measure import regionprops
from cellpose_omni import models
Image.MAX_IMAGE_PIXELS=None
PXUM=1.32
root=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/Katja Myoblasts/E40 hpMb L"
out=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/benchmark"
os.makedirs(out,exist_ok=True)
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
def belly_um(masks):
    ws=[]
    for r in regionprops(masks):
        if r.area<200: continue
        sub=r.image
        d=ndimage.distance_transform_edt(sub); sk=skeletonize(sub)
        w=2*(d[sk].max() if sk.sum()>=3 else d.max())/PXUM
        if w>=3: ws.append(float(w))
    return ws
model=models.CellposeModel(gpu=True, model_type='cyto2_omni')
rows=[]; t0=time.time()
conds=sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
for cond in conds:
    cdir=os.path.join(root,cond)
    mm=re.search(r'(\d+)\s*d',cond); tp=(mm.group(1)+"d") if mm else "?"
    treat=re.sub(r'\s*\d+\s*d','',cond).strip()
    for rf in sorted(glob.glob(os.path.join(cdir,"Result*.txt"))):
        N=re.search(r'Result\s*(\d+)',os.path.basename(rf))
        if not N: continue
        N=int(N.group(1))
        cand=glob.glob(os.path.join(cdir,f"20x_0{N}"))+glob.glob(os.path.join(cdir,f"20x_{N:02d}"))
        if not cand: continue
        ch3=os.path.join(cand[0],"Image_CH3.tif")
        if not os.path.exists(ch3): continue
        gt=parse_um(rf)
        if len(gt)<3: continue
        mhc=np.array(Image.open(ch3).convert("RGB")).astype(np.float32)[:,:,0]
        try:
            masks,_,_=model.eval(gaussian(mhc,1.0),channels=[0,0],omni=True,diameter=60,flow_threshold=0.4)
        except Exception as e:
            print("skip",cond,N,repr(e)[:60]); continue
        masks=np.asarray(masks)
        ws=belly_um(masks)
        if not ws: continue
        rows.append([cond,treat,tp,N,"Omnipose",len(ws),round(float(np.mean(ws)),1),round(float(np.mean(gt)),1),len(gt)])
        print(f"{cond:14s} f{N}: n={len(ws):3d} auto={np.mean(ws):5.1f} katja={np.mean(gt):5.1f} [{time.time()-t0:.0f}s]",flush=True)
with open(out+"/e40_omnipose.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["cond","treat","tp","field","model","n_inst","auto_mean","katja_mean","katja_n"]); w.writerows(rows)
print(f"\nDONE {len(rows)} fields in {time.time()-t0:.0f}s -> e40_omnipose.csv")
