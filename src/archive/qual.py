import numpy as np, os, re, glob, warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw
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
OUT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/for_katja"
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
net=MHNet(40,1).to(dev); net.load_state_dict(torch.load(MD+"/emb_net.pt")); net.eval()
cp=models.CellposeModel(gpu=True)
def instances(mhc,dapi):
    with torch.no_grad(): sem,emb,ori=net(torch.from_numpy(prep_mhc(mhc))[None,None].float().to(dev))
    prob=torch.sigmoid(sem)[0,0].cpu().numpy(); emb=emb[0].cpu().numpy(); ori=F.normalize(ori,dim=1)[0].cpu().numpy()
    mask=binary_closing(remove_small_holes(remove_small_objects(prob>0.45,300),2000),disk(4))
    mhc_s=gaussian(mhc.astype(np.float32),1.5); mask=remove_small_objects(mask&(mhc_s>mhc_s.max()*0.06),300)
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
    return inst, mask, dist
def caliber(instmask,full_dist,R=7):
    sk=skeletonize(instmask); ys,xs=np.where(sk)
    if len(xs)<3: return None
    dvals=full_dist[ys,xs]; i=int(dvals.argmax()); y,x=int(ys[i]),int(xs[i])
    w=2*full_dist[y,x]                                  # BOUNDED (inscribed circle) — cannot span a merge
    pts=np.c_[ys,xs]; tree=cKDTree(pts); idx=tree.query_ball_point([y,x],R)
    if len(idx)<3: perp=np.array([0.0,1.0])
    else:
        nb=pts[idx].astype(float); nb=nb-nb.mean(0)
        vt=np.linalg.svd(nb,full_matrices=False)[2]; perp=np.array([-vt[0][1],vt[0][0]])
    return w,(y,x),perp
def measure(inst,dist,mhcp):
    ws=[]; cals=[]
    for r in regionprops(inst):
        if r.area<200: continue
        if r.eccentricity<0.7 and r.area<1200: continue
        m=inst==r.label
        if mhcp[m].mean()<0.05: continue
        c=caliber(m,dist)
        if c is None or c[0]/PXUM<3: continue
        ws.append(c[0]/PXUM); cals.append((c[1],c[2],c[0]))
    return ws,cals
def draw_ours(mhc,dapi,ws,cals,fusion):
    disp=((mhc-mhc.min())/(mhc.max()-mhc.min()+1e-6)*255).astype(np.uint8)
    db=(np.clip(dapi/np.percentile(dapi,99.5),0,1)*255).astype(np.uint8)
    im=Image.fromarray(np.dstack([disp,np.zeros_like(disp),db])); d=ImageDraw.Draw(im)
    for (pt,perp,w),val in zip(cals,ws):
        y,x=pt; dx,dy=perp[1]*w/2,perp[0]*w/2
        d.line([(x-dx,y-dy),(x+dx,y+dy)],fill=(255,255,0),width=2)
        d.text((int(x)+3,int(y)-6),f"{val:.0f}",fill=(120,255,120))
    d.text((8,8),f"OURS: {len(ws)} fibers | median {np.median(ws):.0f}um | fusion {fusion:.0f}%",fill=(255,255,255))
    return im
def load(fld):
    mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))[:,:,0]
    dapi=np.array(Image.open(fld+"/Image_CH1.tif").convert("RGB"))[:,:,2].astype(np.float32)
    return mhc,dapi
# pick one E40 and one E44 field (held out) that have marked overlays
def pick(base):
    for dp,dn,fn in os.walk(base):
        subs=sorted([d for d in dn if os.path.exists(os.path.join(dp,d,"Image_CH3.tif")) and "20x" in d.lower() and "4x" not in d.lower()])
        for sd in subs:
            fld=os.path.join(dp,sd); mm=glob.glob(fld+"/*measure*.tif")+glob.glob(fld+"/*Overlay m*.tif")
            rf=glob.glob(os.path.dirname(fld)+"/Result*.txt")
            mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))
            if mm and mhc.shape[1]<1900: return fld,mm[0]
    return None
picks=[("E40 hpMb L/scrambled 3d/20x_01", ROOT+"/E40 hpMb L/scrambled 3d/20x_01"),
       pick(ROOT+"/E44 hpMb G Klara")]
rows=[]
print("=== per-field quantitative (bounded caliber) ===")
for item in picks:
    if isinstance(item,tuple) and len(item)==2 and isinstance(item[1],str) and item[1].endswith("20x_01"):
        fld=item[1]; name=item[0]; marked=fld+"/Image_Overlay m.tif"
    else:
        fld,marked=item; name=os.path.relpath(fld,ROOT)
    mhc,dapi=load(fld)
    inst,mask,dist=instances(mhc,dapi)
    nuc,_,_=cp.eval(dapi,flow_threshold=0.4,cellprob_threshold=0.0)
    nc=np.array(ndimage.center_of_mass(np.ones_like(nuc),nuc,range(1,int(nuc.max())+1))).astype(int)
    fusion=100*(mask[nc[:,0],nc[:,1]]).sum()/len(nc)
    ws,cals=measure(inst,dist,prep_mhc(mhc))
    ours=draw_ours(mhc,dapi,ws,cals,fusion)
    katja=Image.open(marked).convert("RGB").resize((mhc.shape[1],mhc.shape[0]))
    ImageDraw.Draw(katja).text((8,8),"KATJA (manual calipers + nuclei)",fill=(255,255,255))
    rows.append((ours,katja,name))
    gt=parse_um(glob.glob(os.path.dirname(fld)+"/Result*.txt")[0]) if glob.glob(os.path.dirname(fld)+"/Result*.txt") else []
    wmax=max(ws) if ws else 0
    print(f"  {name}: ours {len(ws)} fibers, median {np.median(ws):.0f}um, max {wmax:.0f}um (bounded) | fusion {fusion:.0f}%")
W,H=960,720
canvas=Image.new("RGB",(W*2+12,H*2+52),(16,16,16)); d=ImageDraw.Draw(canvas)
for i,(o,kj,name) in enumerate(rows):
    canvas.paste(o.resize((W,H)),(0,i*(H+22)+26)); canvas.paste(kj.resize((W,H)),(W+12,i*(H+22)+26))
    d.text((8,i*(H+22)+6),f"Test field {i+1}: {name}   (LEFT ours, bounded calipers · RIGHT Katja)",fill=(255,255,0))
canvas.save(OUT+"/qualitative_testset.png")
print("saved -> for_katja/qualitative_testset.png")
