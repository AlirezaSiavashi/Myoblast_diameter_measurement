import numpy as np, os, re, glob, warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
from scipy.stats import pearsonr
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, binary_closing, disk, remove_small_holes
from skimage.measure import label as sklabel, regionprops
Image.MAX_IMAGE_PIXELS=None
dev="cuda"; PXUM=1.32; EMB=8
ROOT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/Katja Myoblasts"
MD=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/models"
def prep_mhc(g):
    g=g.astype(np.float32); bg=gaussian(g,40); fg=np.clip(g-bg,0,None)
    lo,hi=np.percentile(fg,[1,99.5]); s=np.clip((fg-lo)/(hi-lo+1e-6),0,1)
    return (0.5*s+0.5*(s**0.6)).astype(np.float32)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
class MHNet(nn.Module):
    def __init__(s,c=40,inp=1):
        super().__init__(); s.e1=cbr(inp,c);s.e2=cbr(c,c*2);s.e3=cbr(c*2,c*4);s.e4=cbr(c*4,c*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(c*8,c*4,2,2);s.d3=cbr(c*8,c*4);s.u2=nn.ConvTranspose2d(c*4,c*2,2,2);s.d2=cbr(c*4,c*2)
        s.u1=nn.ConvTranspose2d(c*2,c,2,2);s.d1=cbr(c*2,c);s.sem=nn.Conv2d(c,1,1);s.emb=nn.Conv2d(c,EMB,1);s.ori=nn.Conv2d(c,2,1)
    def forward(s,x):
        e1=s.e1(x);e2=s.e2(s.p(e1));e3=s.e3(s.p(e2));e4=s.e4(s.p(e3))
        d3=s.d3(torch.cat([s.u3(e4),e3],1));d2=s.d2(torch.cat([s.u2(d3),e2],1));f=s.d1(torch.cat([s.u1(d2),e1],1))
        return s.sem(f),s.emb(f),s.ori(f)
def natkey(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)',s)]
def fields_marked(base):
    out=[]
    for dp,dn,fn in os.walk(base):
        subs=sorted([d for d in dn if os.path.exists(os.path.join(dp,d,"Image_CH3.tif")) and "20x" in d.lower() and "4x" not in d.lower()],key=natkey)
        for sd in subs:
            fld=os.path.join(dp,sd)
            m=glob.glob(fld+"/*measure*.tif")+glob.glob(fld+"/*Overlay m*.tif")+glob.glob(fld+"/*Overlay  m*.tif")
            if m and os.path.exists(fld+"/Image_Overlay.tif"): out.append((fld,m[0]))
    return out
def her_calipers(clean,mark):
    diff=np.abs(mark.astype(int)-clean.astype(int)).max(2); mk=ndimage.binary_closing(diff>40,iterations=1); res=[]
    for r in regionprops(sklabel(mk)):
        if r.area<40 or r.eccentricity<0.70: continue
        ys,xs=r.coords[:,0],r.coords[:,1]; col=mark[ys,xs].astype(int).mean(0)
        if col[1]>col[0]+25 and col[1]>col[2]+25: continue
        c=r.coords.astype(float); c0=c-c.mean(0)
        try:
            vt=np.linalg.svd(c0,full_matrices=False)[2]; d=vt[0]; L=float(np.ptp(c0@d))
        except: continue
        cy,cx=r.centroid
        res.append((cx,cy,float(d[1]),float(d[0]),max(L-2,3)))   # cx,cy, dir(dx,dy), her_width_px
    return res
net=MHNet(40,1).to(dev); net.load_state_dict(torch.load(MD+"/emb_net.pt")); net.eval()
def our_mask(mhc):
    with torch.no_grad(): sem,_,_=net(torch.from_numpy(prep_mhc(mhc))[None,None].float().to(dev))
    prob=torch.sigmoid(sem)[0,0].cpu().numpy()
    mask=binary_closing(remove_small_holes(remove_small_objects(prob>0.45,300),2000),disk(4))
    mhc_s=gaussian(mhc.astype(np.float32),1.5); mask=remove_small_objects(mask&(mhc_s>mhc_s.max()*0.06),300)
    return mask
def ray(mask,cy,cx,dy,dx,H,W,maxlen=120):
    for t in range(1,maxlen):
        yy=int(round(cy+dy*t)); xx=int(round(cx+dx*t))
        if yy<0 or yy>=H or xx<0 or xx>=W or not mask[yy,xx]: return t-1
    return maxlen
for exp in ["E40 hpMb L","E44 hpMb G Klara"]:
    hw=[]; ow=[]; owdt=[]; bnd=[]; miss=0; tot=0
    for fld,marked in fields_marked(os.path.join(ROOT,exp)):
        mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))[:,:,0]
        if mhc.shape[1]>=1900: continue
        clean=np.array(Image.open(fld+"/Image_Overlay.tif").convert("RGB")); mk=np.array(Image.open(marked).convert("RGB"))
        if clean.shape!=mk.shape: continue
        cals=her_calipers(clean,mk)
        if len(cals)<3: continue
        m=our_mask(mhc); H,W=m.shape; dist=ndimage.distance_transform_edt(m)
        for cx,cy,dx,dy,hwid in cals:
            tot+=1; cyi,cxi=int(round(cy)),int(round(cx))
            if not (0<=cyi<H and 0<=cxi<W and m[cyi,cxi]):
                found=False
                for rad in range(1,7):
                    ys,xs=np.where(m[max(0,cyi-rad):cyi+rad+1,max(0,cxi-rad):cxi+rad+1])
                    if len(xs): cyi=max(0,cyi-rad)+ys[0]; cxi=max(0,cxi-rad)+xs[0]; found=True; break
                if not found: miss+=1; continue
            a=ray(m,cyi,cxi,dy,dx,H,W); b=ray(m,cyi,cxi,-dy,-dx,H,W)   # cast along HER direction
            hh=hwid/2.0
            # bounded width: cap the runaway using the inscribed circle (2*dist), which cannot span a merge
            our_w=min(a+b+1, 2.6*2*dist[cyi,cxi])
            our_w_dt=2*dist[cyi,cxi]
            hw.append(hwid/PXUM); ow.append(our_w/PXUM); owdt.append(our_w_dt/PXUM)
            bnd.append(min(abs(a-hh),2.6*dist[cyi,cxi])/PXUM); bnd.append(min(abs(b-hh),2.6*dist[cyi,cxi])/PXUM)
    hw=np.array(hw); ow=np.array(ow); owdt=np.array(owdt); bnd=np.array(bnd)
    print(f"\n=== {exp}: at Katja's caliper points, ALONG her direction ({len(hw)} calipers, {100*miss/tot:.0f}% missed) ===")
    print(f"  WIDTH (bounded ray):  r={pearsonr(ow,hw)[0]:.3f} ratio={np.mean(ow/hw):.2f} median-ratio={np.median(ow/hw):.2f} within5um={100*np.mean(np.abs(ow-hw)<=5):.0f}%")
    print(f"  WIDTH (dist-transform): r={pearsonr(owdt,hw)[0]:.3f} ratio={np.mean(owdt/hw):.2f} within5um={100*np.mean(np.abs(owdt-hw)<=5):.0f}%")
    print(f"  EDGE HD95(capped)={np.percentile(bnd,95):.1f}um mean={bnd.mean():.1f}um median={np.median(bnd):.1f}um")
