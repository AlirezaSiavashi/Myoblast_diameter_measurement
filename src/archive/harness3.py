import numpy as np, os, re, glob
from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops
from scipy.stats import pearsonr
Image.MAX_IMAGE_PIXELS=None
ROOT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/Katja Myoblasts"
OUT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/caliper_harness"
PXUM=1.32; CAP=4.0
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
                    except:pass
            return v
        except:continue
    return v
def major_len(r):
    c=r.coords.astype(float); c-=c.mean(0)
    if len(c)<2: return 0.0
    _,_,vt=np.linalg.svd(c,full_matrices=False)
    proj=c@vt[0]; return float(proj.max()-proj.min())
def detect(clean,mark,N):
    diff=np.abs(mark.astype(int)-clean.astype(int)).max(2)
    mk=ndimage.binary_closing(diff>40,iterations=1)
    cands=[]
    for r in regionprops(label(mk)):
        if r.area<40: continue
        ys,xs=r.coords[:,0],r.coords[:,1]; col=mark[ys,xs].astype(int).mean(0)
        if col[1]>col[0]+25 and col[1]>col[2]+25: continue        # green number
        cands.append((r.eccentricity,major_len(r),(xs.mean(),ys.mean())))
    cands.sort(key=lambda t:-t[0])            # most elongated first (calipers >> x-marks)
    sel=cands[:N]                              # count constraint from Result.txt
    return [max((L-CAP)/PXUM,1.0) for _,L,_ in sel]

tasks=[]
for dp,dn,fn in os.walk(ROOT):
    res=[f for f in fn if re.match(r'result',f,re.I) and f.lower().endswith('.txt')]
    subs=[d for d in dn if os.path.exists(os.path.join(dp,d,"Image_Overlay.tif"))]
    if res and subs:
        res.sort(key=natkey); subs.sort(key=natkey)
        for rf,sd in zip(res,subs): tasks.append((dp,rf,sd))

pa=[];pg=[];fa=[];fg=[]
for dp,rf,sd in tasks:
    field=os.path.join(dp,sd)
    marked=glob.glob(os.path.join(field,"*measure*.tif"))+glob.glob(os.path.join(field,"*Overlay m*.tif"))+glob.glob(os.path.join(field,"*Overlay  m*.tif"))
    cp=os.path.join(field,"Image_Overlay.tif")
    if not marked or not os.path.exists(cp): continue
    gt=parse_um(os.path.join(dp,rf))
    if len(gt)<3: continue
    try:
        clean=np.array(Image.open(cp).convert("RGB")); mark=np.array(Image.open(marked[0]).convert("RGB"))
        if clean.shape!=mark.shape: continue
        au=detect(clean,mark,len(gt))
    except: continue
    if len(au)<3: continue
    k=min(len(au),len(gt))
    aus=sorted(au,reverse=True)[:k]; gts=sorted(gt,reverse=True)[:k]
    pa+=aus; pg+=gts
    fa.append(np.mean(au)); fg.append(np.mean(gt))
pa=np.array(pa);pg=np.array(pg);fa=np.array(fa);fg=np.array(fg)
A=np.polyfit(pa,pg,1); pred=np.polyval(A,pa); R2=1-np.sum((pg-pred)**2)/np.sum((pg-pg.mean())**2)
print(f"fields={len(fa)} | paired labels={len(pa)}")
print(f"PER-LABEL (rank-matched):  r={pearsonr(pa,pg)[0]:.3f}  slope={A[0]:.2f} intercept={A[1]:.1f}  R2={R2:.3f}  median|resid|={np.median(np.abs(pg-pred)):.1f}um")
print(f"PER-FIELD mean width:      r={pearsonr(fa,fg)[0]:.3f}  bias={np.mean(fa-fg):+.1f}um  within5um={100*np.mean(np.abs(fa-fg)<=5):.0f}%  within3um={100*np.mean(np.abs(fa-fg)<=3):.0f}%")
print(f"  auto mean {fa.mean():.1f} vs manual {fg.mean():.1f}um")
print(f"[baseline was: per-field r=0.51, bias +4.2um, within5um=40%]")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].scatter(pg,pa,s=4,alpha=.15); lim=[0,90]; ax[0].plot(lim,lim,'k--');ax[0].set_xlim(lim);ax[0].set_ylim(lim)
ax[0].set_xlabel("Katja width (µm)");ax[0].set_ylabel("auto width (µm)");ax[0].set_title(f"Per-label refined: r={pearsonr(pa,pg)[0]:.2f}, n={len(pa)}")
ax[1].scatter(fg,fa,s=10,alpha=.4); lim2=[0,max(fa.max(),fg.max())+3];ax[1].plot(lim2,lim2,'k--');ax[1].set_xlim(lim2);ax[1].set_ylim(lim2)
ax[1].set_xlabel("manual mean/field (µm)");ax[1].set_ylabel("auto mean/field (µm)");ax[1].set_title(f"Per-field refined: r={pearsonr(fa,fg)[0]:.2f}")
fig.tight_layout(); fig.savefig(OUT+"/caliper_refined.png",dpi=120); print("saved")
