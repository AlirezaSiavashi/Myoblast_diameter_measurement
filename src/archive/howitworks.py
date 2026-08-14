import numpy as np, os, re, glob, warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import convolve, binary_dilation
from scipy.spatial import cKDTree
from skimage.filters import gaussian
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk, remove_small_holes
from skimage.measure import label as sklabel, regionprops
from skimage.segmentation import watershed
from skimage.color import label2rgb
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
K8=np.ones((3,3),int); K8[1,1]=0
def prune_spurs(skel,min_len=9):
    sk=skel.copy()
    for _ in range(25):
        nb=convolve(sk.astype(int),K8,mode='constant')
        endpoints=sk&(nb==1); branch=binary_dilation(sk&(nb>=3),iterations=1)
        seg=sklabel(sk&~branch,connectivity=2); removed=False
        for r in regionprops(seg):
            ys,xs=r.coords[:,0],r.coords[:,1]
            if endpoints[ys,xs].any() and r.area<min_len: sk[ys,xs]=False; removed=True
        if not removed: break
    return sk
net=MHNet(40,1).to(dev); net.load_state_dict(torch.load(MD+"/emb_net.pt")); net.eval()
cp=models.CellposeModel(gpu=True)
def instances(mhc,dapi):
    with torch.no_grad(): sem,emb,ori=net(torch.from_numpy(prep_mhc(mhc))[None,None].float().to(dev))
    prob=torch.sigmoid(sem)[0,0].cpu().numpy(); emb=emb[0].cpu().numpy(); ori=F.normalize(ori,dim=1)[0].cpu().numpy()
    mask=binary_closing(remove_small_holes(remove_small_objects(prob>0.45,300),2000),disk(4))
    mhc_s=gaussian(mhc.astype(np.float32),1.5); mask=remove_small_objects(mask&(mhc_s>mhc_s.max()*0.06),300)
    nuc,_,_=cp.eval(dapi,flow_threshold=0.4,cellprob_threshold=0.0)
    dist=ndimage.distance_transform_edt(mask); sk=prune_spurs(skeletonize(mask),9)
    nb=convolve(sk.astype(int),K8,mode='constant'); br=binary_dilation(sk&(nb>=3),iterations=1)
    seeds=sklabel(sk&~br,connectivity=2)
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
    mO={i:(lambda o:o/(np.linalg.norm(o)+1e-6))(ori[:,frag==i].mean(1)) for i in ids}
    for i in ids:
        di=binary_dilation(frag==i,iterations=2)
        for j in np.unique(frag[di&(frag!=i)&(frag>0)]):
            j=int(j)
            if j<=i: continue
            if np.dot(mO[i],mO[j])>-0.35: par[find(i)]=find(j)
    inst=np.zeros_like(frag); remap={}; nx=1
    for i in ids:
        r=find(i)
        if r not in remap: remap[r]=nx; nx+=1
        inst[frag==i]=remap[r]
    return inst, mask, dist
def caliber(instmask,full_dist,R=7):
    sk=skeletonize(instmask); ys,xs=np.where(sk)
    if len(xs)<3: return None
    dv=full_dist[ys,xs]; i=int(dv.argmax()); y,x=int(ys[i]),int(xs[i]); w=2*full_dist[y,x]
    pts=np.c_[ys,xs]; tree=cKDTree(pts); idx=tree.query_ball_point([y,x],R)
    if len(idx)<3: perp=np.array([0.0,1.0])
    else:
        nb=pts[idx].astype(float); nb=nb-nb.mean(0); vt=np.linalg.svd(nb,full_matrices=False)[2]; perp=np.array([-vt[0][1],vt[0][0]])
    return w,(y,x),perp,sk
def load(fld):
    mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))[:,:,0]
    dapi=np.array(Image.open(fld+"/Image_CH1.tif").convert("RGB"))[:,:,2].astype(np.float32)
    return mhc,dapi
picks=["E40 hpMb L/scrambled 3d/20x_01","E40 hpMb L/scrambled 5d/20x_01","E40 hpMb L/KIF3_20x... "]
picks=["E40 hpMb L/scrambled 3d/20x_01","E40 hpMb L/scrambled 5d/20x_01","E40 hpMb L/mi133 3d/20x_02"]
panels=[]
for name in picks:
    fld=ROOT+"/"+name
    mhc,dapi=load(fld); inst,mask,dist=instances(mhc,dapi)
    disp=((mhc-mhc.min())/(mhc.max()-mhc.min()+1e-6)*255).astype(np.uint8); db=(np.clip(dapi/np.percentile(dapi,99.5),0,1)*255).astype(np.uint8)
    base=np.dstack([disp,np.zeros_like(disp),db]).astype(np.float32)/255.
    col=(label2rgb(inst,image=base,bg_label=0,alpha=0.32,image_alpha=0.68)*255).astype(np.uint8)
    im=Image.fromarray(col); d=ImageDraw.Draw(im); diam=[]
    for r in regionprops(inst):
        if r.area<200: continue
        if r.eccentricity<0.7 and r.area<1200: continue
        m=inst==r.label
        if prep_mhc(mhc)[m].mean()<0.05: continue
        c=caliber(m,dist)
        if c is None or c[0]/PXUM<3: continue
        w,(y,x),perp,sk=c
        ys,xs=np.where(sk)                                   # 1) centerline (cyan)
        for yy,xx in zip(ys,xs): d.point((xx,yy),fill=(0,220,255))
        dx,dy=perp[1]*w/2,perp[0]*w/2                        # 3) perpendicular caliper (yellow)
        d.line([(x-dx,y-dy),(x+dx,y+dy)],fill=(255,255,0),width=2)
        d.ellipse([x-3,y-3,x+3,y+3],fill=(255,255,255))      # 2) thickest point (white dot)
        d.text((int(x)+4,int(y)-7),f"{w/PXUM:.0f}",fill=(140,255,140))
        diam.append(w/PXUM)
    d.text((8,8),f"{name.split('/')[-2]}: {len(diam)} fibers | median {np.median(diam):.0f} um",fill=(255,255,255))
    panels.append((im,name,sorted([round(x) for x in diam],reverse=True)))
    print(f"\n=== {name} ===")
    print(f"  {len(diam)} fibers | diameters (um): {sorted([round(x) for x in diam],reverse=True)}")
    print(f"  median {np.median(diam):.1f} um | mean {np.mean(diam):.1f} um | range {min(diam):.0f}-{max(diam):.0f}")
# compose 3 panels stacked
W,H=960,720
cv=Image.new("RGB",(W,H*3+80),(16,16,16)); d=ImageDraw.Draw(cv)
d.text((10,6),"HOW THE DIAMETER IS DETERMINED  —  cyan = centerline · white dot = thickest point · yellow = perpendicular width",fill=(230,230,230))
for i,(im,name,dl) in enumerate(panels):
    cv.paste(im,(0,i*(H+18)+24))
cv.save(OUT+"/how_diameter_works.png")
print("\nsaved -> for_katja/how_diameter_works.png")
