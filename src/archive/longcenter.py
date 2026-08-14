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
from cellpose import models
Image.MAX_IMAGE_PIXELS=None
dev="cuda"; PXUM=1.32
ROOT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/Katja Myoblasts"
MD=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/models"
OUT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/for_katja"
MINLEN=int(os.environ.get("MINLEN","28"))   # min centerline length (px) to count as a real fiber section
def prep_mhc(g):
    g=g.astype(np.float32); bg=gaussian(g,40); fg=np.clip(g-bg,0,None)
    lo,hi=np.percentile(fg,[1,99.5]); s=np.clip((fg-lo)/(hi-lo+1e-6),0,1)
    return (0.5*s+0.5*(s**0.6)).astype(np.float32)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
class MHNet(nn.Module):
    def __init__(s,c=40,inp=1):
        super().__init__(); s.e1=cbr(inp,c);s.e2=cbr(c,c*2);s.e3=cbr(c*2,c*4);s.e4=cbr(c*4,c*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(c*8,c*4,2,2);s.d3=cbr(c*8,c*4);s.u2=nn.ConvTranspose2d(c*4,c*2,2,2);s.d2=cbr(c*4,c*2)
        s.u1=nn.ConvTranspose2d(c*2,c,2,2);s.d1=cbr(c*2,c);s.sem=nn.Conv2d(c,1,1);s.emb=nn.Conv2d(c,8,1);s.ori=nn.Conv2d(c,2,1)
    def forward(s,x):
        e1=s.e1(x);e2=s.e2(s.p(e1));e3=s.e3(s.p(e2));e4=s.e4(s.p(e3))
        d3=s.d3(torch.cat([s.u3(e4),e3],1));d2=s.d2(torch.cat([s.u2(d3),e2],1));f=s.d1(torch.cat([s.u1(d2),e1],1))
        return s.sem(f),s.emb(f),s.ori(f)
K8=np.ones((3,3),int); K8[1,1]=0
def prune_spurs(skel,min_len=9):
    sk=skel.copy()
    for _ in range(25):
        nb=convolve(sk.astype(int),K8,mode='constant')
        ep=sk&(nb==1); br=binary_dilation(sk&(nb>=3),iterations=1); seg=sklabel(sk&~br,connectivity=2); rem=False
        for r in regionprops(seg):
            ys,xs=r.coords[:,0],r.coords[:,1]
            if ep[ys,xs].any() and r.area<min_len: sk[ys,xs]=False; rem=True
        if not rem: break
    return sk
net=MHNet(40,1).to(dev); net.load_state_dict(torch.load(MD+"/emb_net.pt")); net.eval()
cp=models.CellposeModel(gpu=True)
def semseg(mhc):
    with torch.no_grad(): sem,_,_=net(torch.from_numpy(prep_mhc(mhc))[None,None].float().to(dev))
    prob=torch.sigmoid(sem)[0,0].cpu().numpy()
    mask=binary_closing(remove_small_holes(remove_small_objects(prob>0.45,300),2000),disk(4))
    mhc_s=gaussian(mhc.astype(np.float32),1.5)
    return remove_small_objects(mask&(mhc_s>mhc_s.max()*0.06),300)
from collections import deque
def long_centerlines(mask):
    dist=ndimage.distance_transform_edt(mask)
    lab=sklabel(mask); out=[]
    for rr in regionprops(lab):
        cc=lab==rr.label
        sk=prune_spurs(skeletonize(cc),9)
        ys,xs=np.where(sk)
        if len(xs)<MINLEN: continue
        S=set(zip(ys.tolist(),xs.tolist())); start=(int(ys[0]),int(xs[0]))
        def neigh(p):
            y,x=p
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    if (dy or dx) and (y+dy,x+dx) in S: yield (y+dy,x+dx)
        def bfs(s):
            dd={s:0}; par={s:None}; q=deque([s])
            while q:
                u=q.popleft()
                for v in neigh(u):
                    if v not in dd: dd[v]=dd[u]+1; par[v]=u; q.append(v)
            f=max(dd,key=dd.get); return f,par
        A,_=bfs(start); B,par=bfs(A)                        # double-BFS -> longest geodesic path (main spine)
        path=[]; cur=B
        while cur is not None: path.append(cur); cur=par[cur]
        if len(path)<MINLEN: continue
        pw=[2*dist[y,x]/PXUM for (y,x) in path]
        i=int(np.argmax(pw)); y,x=path[i]; w=pw[i]          # thickest point ON the main spine (fans ignored)
        if w<3: continue
        seg=np.array(path[max(0,i-6):i+7],float)
        if len(seg)>=3:
            c0=seg-seg.mean(0); vt=np.linalg.svd(c0,full_matrices=False)[2]; perp=np.array([-vt[0][1],vt[0][0]])
        else: perp=np.array([0.0,1.0])
        out.append((w,(y,x),perp,np.array(path)))
    return out,dist
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
picks=["E40 hpMb L/scrambled 3d/20x_01","E40 hpMb L/scrambled 5d/20x_01","E40 hpMb L/mi133 3d/20x_02"]
gtmap={"scrambled 3d/20x_01":"scrambled 3d/Result 1.txt"}
panels=[]
for name in picks:
    fld=ROOT+"/"+name
    mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))[:,:,0]
    dapi=np.array(Image.open(fld+"/Image_CH1.tif").convert("RGB"))[:,:,2]
    mask=semseg(mhc); cls,dist=long_centerlines(mask)
    disp=((mhc-mhc.min())/(mhc.max()-mhc.min()+1e-6)*255).astype(np.uint8); db=(np.clip(dapi/np.percentile(dapi,99.5),0,1)*255).astype(np.uint8)
    im=Image.fromarray(np.dstack([disp,np.zeros_like(disp),db.astype(np.uint8)])); d=ImageDraw.Draw(im); diam=[]
    for w,(y,x),perp,coords in cls:
        for yy,xx in coords: d.point((xx,yy),fill=(0,230,255))          # long centerline (cyan)
        dxp,dyp=perp[1]*w*PXUM/2,perp[0]*w*PXUM/2
        d.line([(x-dxp,y-dyp),(x+dxp,y+dyp)],fill=(255,255,0),width=2)  # perpendicular caliper (yellow)
        d.ellipse([x-3,y-3,x+3,y+3],fill=(255,255,255))                 # thickest point
        d.text((int(x)+4,int(y)-7),f"{w:.0f}",fill=(150,255,150)); diam.append(w)
    d.text((8,8),f"{name.split('/')[-2]}: {len(diam)} long centerlines | median {np.median(diam):.0f} um",fill=(255,255,255))
    panels.append(im)
    print(f"\n=== {name} (MINLEN={MINLEN}px) ===")
    print(f"  {len(diam)} long centerlines | diameters(um): {sorted([round(x) for x in diam],reverse=True)}")
    print(f"  median {np.median(diam):.1f} | mean {np.mean(diam):.1f}")
    key=name.split("L/")[-1]
    rf=glob.glob(os.path.dirname(fld)+"/Result*.txt")
    if rf:
        gt=parse_um(rf[0])
        if gt: print(f"  KATJA same field: {len(gt)} calipers, median {np.median(gt):.1f}, values {sorted([round(x) for x in gt],reverse=True)}")
W,H=960,720
cv=Image.new("RGB",(W,H*3+30),(16,16,16)); dd=ImageDraw.Draw(cv)
dd.text((10,6),f"LONG-CENTERLINE DIAMETER (MINLEN={MINLEN}px)  —  cyan = kept long centerline · yellow = thickest perpendicular width · end-fans ignored",fill=(230,230,230))
for i,im in enumerate(panels): cv.paste(im,(0,i*(H+6)+22))
cv.save(OUT+"/long_centerline.png"); print("\nsaved -> for_katja/long_centerline.png")
