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
def norm(g):
    g=g.astype(np.float32); lo,hi=np.percentile(g,[1,99.5]); return np.clip((g-lo)/(hi-lo+1e-6),0,1).astype(np.float32)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
class MHNet(nn.Module):
    def __init__(s,c=40):
        super().__init__(); s.e1=cbr(3,c);s.e2=cbr(c,c*2);s.e3=cbr(c*2,c*4);s.e4=cbr(c*4,c*8);s.p=nn.MaxPool2d(2)
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
            marked=glob.glob(fld+"/*measure*.tif")+glob.glob(fld+"/*Overlay m*.tif")+glob.glob(fld+"/*Overlay  m*.tif")
            if marked and os.path.exists(fld+"/Image_Overlay.tif"): out.append((fld,marked[0]))
    return out
def her_calipers(clean,mark):
    diff=np.abs(mark.astype(int)-clean.astype(int)).max(2)
    mk=ndimage.binary_closing(diff>40,iterations=1); res=[]
    for r in regionprops(sklabel(mk)):
        if r.area<40 or r.eccentricity<0.70: continue
        ys,xs=r.coords[:,0],r.coords[:,1]; col=mark[ys,xs].astype(int).mean(0)
        if col[1]>col[0]+25 and col[1]>col[2]+25: continue
        c=r.coords.astype(float); c=c-c.mean(0)
        try: L=float((c@np.linalg.svd(c,full_matrices=False)[2][0]).ptp())
        except: L=float(max(r.image.shape))
        cy,cx=r.centroid
        res.append((int(cx),int(cy),max(L-2,3)/PXUM))   # her x,y, her width(um)
    return res
net=MHNet(40).to(dev); net.load_state_dict(torch.load(MD+"/mc_net.pt")); net.eval()
def our_mask(mhc,dapi,myog):
    x3=np.stack([prep_mhc(mhc),norm(dapi),norm(myog)]).astype(np.float32)
    with torch.no_grad(): sem,_,_=net(torch.from_numpy(x3)[None].float().to(dev))
    prob=torch.sigmoid(sem)[0,0].cpu().numpy()
    return binary_closing(remove_small_holes(remove_small_objects(prob>0.45,300),2000),disk(4))

for exp in ["E40 hpMb L","E44 hpMb G Klara"]:
    hers=[]; ours=[]; missed=0; tot=0
    for fld,marked in fields_marked(os.path.join(ROOT,exp)):
        mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))[:,:,0]
        if mhc.shape[1]>=1900: continue
        clean=np.array(Image.open(fld+"/Image_Overlay.tif").convert("RGB"))
        mark=np.array(Image.open(marked).convert("RGB"))
        if clean.shape!=mark.shape: continue
        dapi=np.array(Image.open(fld+"/Image_CH1.tif").convert("RGB"))[:,:,2].astype(np.float32)
        myog=np.array(Image.open(fld+"/Image_CH2.tif").convert("RGB"))[:,:,1] if os.path.exists(fld+"/Image_CH2.tif") else np.zeros_like(mhc)
        cals=her_calipers(clean,mark)
        if len(cals)<3: continue
        m=our_mask(mhc,dapi,myog); dist=ndimage.distance_transform_edt(m); H,W=m.shape
        for cx,cy,hw in cals:
            tot+=1
            y0,y1,x0,x1=max(0,cy-6),min(H,cy+7),max(0,cx-6),min(W,cx+7)
            local=dist[y0:y1,x0:x1]
            if local.max()<1:            # our mask absent here -> we missed this fiber
                missed+=1; continue
            ow=2*local.max()/PXUM
            hers.append(hw); ours.append(ow)
    hers=np.array(hers); ours=np.array(ours)
    print(f"\n=== {exp}: LIKE-FOR-LIKE at Katja's own caliper locations ===")
    print(f"  Katja calipers: {tot} | our fiber detected at: {tot-missed} ({100*(tot-missed)/tot:.0f}%) | missed: {missed}")
    if len(hers)>3:
        print(f"  width agreement (same fibers): r={pearsonr(ours,hers)[0]:.3f} bias={np.mean(ours-hers):+.1f}um ratio={np.mean(ours/hers):.2f} within5um={100*np.mean(np.abs(ours-hers)<=5):.0f}%")
        print(f"  mean our {ours.mean():.1f}um vs Katja {hers.mean():.1f}um")
