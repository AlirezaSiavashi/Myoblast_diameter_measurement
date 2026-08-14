import numpy as np, os, re, glob, warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import convolve, binary_dilation
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from skimage.filters import gaussian
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk, remove_small_holes
from skimage.measure import label as sklabel, regionprops
from skimage.segmentation import watershed
from cellpose import models
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
def parse_um(fp):
    v=[]
    for enc in ("utf-8","latin-1"):
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
def natkey(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)',s)]
def fields_under(base):
    out=[]
    for dp,dn,fn in os.walk(base):
        res=sorted([f for f in fn if re.match(r'result',f,re.I) and f.lower().endswith('.txt')],key=natkey)
        subs=sorted([d for d in dn if os.path.exists(os.path.join(dp,d,"Image_CH3.tif")) and "20x" in d.lower() and "4x" not in d.lower() and "opa" not in d.lower()],key=natkey)
        for rf,sd in zip(res,subs): out.append((os.path.join(dp,rf),os.path.join(dp,sd)))
    return out
net=MHNet(40,inp=1).to(dev); net.load_state_dict(torch.load(MD+"/emb_net.pt")); net.eval()
cp=models.CellposeModel(gpu=True)
def instances(mhc,dapi,mhc_gate=True):
    pre=prep_mhc(mhc)
    with torch.no_grad(): sem,emb,ori=net(torch.from_numpy(pre)[None,None].float().to(dev))
    prob=torch.sigmoid(sem)[0,0].cpu().numpy(); emb=emb[0].cpu().numpy(); ori=F.normalize(ori,dim=1)[0].cpu().numpy()
    mask=binary_closing(remove_small_holes(remove_small_objects(prob>0.45,300),2000),disk(4))
    if mhc_gate:
        mhc_s=gaussian(mhc.astype(np.float32),1.5)
        mask=mask & (mhc_s>mhc_s.max()*0.06)      # require MHC support -> kills nuclei-only false positives
        mask=remove_small_objects(mask,300)
    nuc,_,_=cp.eval(dapi,flow_threshold=0.4,cellprob_threshold=0.0)
    dist=ndimage.distance_transform_edt(mask); sk=skeletonize(mask)
    k=np.ones((3,3),int); k[1,1]=0; nb=convolve(sk.astype(int),k,mode='constant')
    br=binary_dilation(sk&(nb>=3),iterations=1); seeds=sklabel(sk&~br,connectivity=2)
    for r in regionprops(seeds):
        if r.area<6: seeds[seeds==r.label]=0
    seeds=sklabel(seeds>0,connectivity=2); frag=remove_small_objects(watershed(-dist,markers=seeds,mask=mask),150)
    ids=[r.label for r in regionprops(frag)]; par={i:i for i in ids}
    def find(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    for nid in range(1,int(nuc.max())+1):
        fr=np.unique(frag[nuc==nid]); fr=fr[fr>0]
        for j in fr[1:]:
            if int(fr[0]) in par and int(j) in par: par[find(int(fr[0]))]=find(int(j))
    mE={i:emb[:,frag==i].mean(1) for i in ids}; mO={i:(lambda o:o/(np.linalg.norm(o)+1e-6))(ori[:,frag==i].mean(1)) for i in ids}
    for i in ids:
        di=binary_dilation(frag==i,iterations=2)
        for j in np.unique(frag[di&(frag!=i)&(frag>0)]):
            j=int(j)
            if j<=i: continue
            if np.linalg.norm(mE[i]-mE[j])<1.0 and np.dot(mO[i],mO[j])>0: par[find(i)]=find(j)
    inst=np.zeros_like(frag); remap={}; nx=1
    for i in ids:
        r=find(i)
        if r not in remap: remap[r]=nx; nx+=1
        inst[frag==i]=remap[r]
    return inst, mask
def caliber(instmask,full_dist,R=7):
    sk=skeletonize(instmask); ys,xs=np.where(sk)
    if len(xs)<3:
        d=full_dist[instmask]; return (2*float(d.max()) if d.size else 0.0)
    dvals=full_dist[ys,xs]; i=int(dvals.argmax()); y,x=int(ys[i]),int(xs[i])
    return 2*full_dist[y,x]
def measure(inst,mask,mhcp):
    fd=ndimage.distance_transform_edt(mask); ws=[]
    for r in regionprops(inst):
        if r.area<200: continue
        if r.eccentricity<0.7 and r.area<1200: continue
        m=inst==r.label
        if mhcp[m].mean()<0.05: continue
        w=caliber(m,fd)
        if w/PXUM>=3: ws.append(w/PXUM)
    return ws
for exp in ["E40 hpMb L","E44 hpMb G Klara"]:
    auto=[]; km=[]; nn_=[]
    for rf,fld in fields_under(os.path.join(ROOT,exp)):
        gt=parse_um(rf)
        if len(gt)<3: continue
        mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))[:,:,0]
        if mhc.shape[1]>=1900: continue
        dapi=np.array(Image.open(fld+"/Image_CH1.tif").convert("RGB"))[:,:,2].astype(np.float32)
        inst,mask=instances(mhc,dapi); ws=measure(inst,mask,prep_mhc(mhc))
        if len(ws)<3: continue
        auto.append(np.mean(ws)); km.append(np.mean(gt)); nn_.append(len(ws))
    auto=np.array(auto); km=np.array(km)
    print(f"=== {exp}: MHC-ONLY model + MHC-gate ({len(auto)} fields) ===")
    print(f"  per-fiber diam r={pearsonr(auto,km)[0]:.3f} bias={np.mean(auto-km):+.1f} ratio={np.mean(auto/km):.2f} within5um={100*np.mean(np.abs(auto-km)<=5):.0f}% | {np.mean(nn_):.0f} fibers/field")
