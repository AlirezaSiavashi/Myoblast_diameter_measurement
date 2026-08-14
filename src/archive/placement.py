import numpy as np, os, glob, warnings
from collections import deque
warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import convolve, binary_dilation
from skimage.filters import gaussian
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk, remove_small_holes
from skimage.measure import label as sklabel, regionprops
Image.MAX_IMAGE_PIXELS=None
dev="cuda"; PXUM=1.32; MINLEN=28
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
def semseg(mhc):
    with torch.no_grad(): sem,_,_=net(torch.from_numpy(prep_mhc(mhc))[None,None].float().to(dev))
    prob=torch.sigmoid(sem)[0,0].cpu().numpy()
    mask=binary_closing(remove_small_holes(remove_small_objects(prob>0.45,300),2000),disk(4))
    mhc_s=gaussian(mhc.astype(np.float32),1.5)
    return remove_small_objects(mask&(mhc_s>mhc_s.max()*0.06),300)
def centerlines(mask):
    dist=ndimage.distance_transform_edt(mask); lab=sklabel(mask); out=[]
    for rr in regionprops(lab):
        cc=lab==rr.label; sk=prune_spurs(skeletonize(cc),9); ys,xs=np.where(sk)
        if len(xs)<MINLEN: continue
        S=set(zip(ys.tolist(),xs.tolist())); start=(int(ys[0]),int(xs[0]))
        def neigh(p):
            y,x=p
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    if (dy or dx) and (y+dy,x+dx) in S: yield (y+dy,x+dx)
        def bfs(s):
            dd={s:0};par={s:None};q=deque([s])
            while q:
                u=q.popleft()
                for v in neigh(u):
                    if v not in dd: dd[v]=dd[u]+1;par[v]=u;q.append(v)
            return max(dd,key=dd.get),par
        A,_=bfs(start); B,par=bfs(A); path=[]; cur=B
        while cur is not None: path.append(cur); cur=par[cur]
        if len(path)<MINLEN: continue
        pw=[2*dist[y,x]/PXUM for (y,x) in path]; i=int(np.argmax(pw)); y,x=path[i]; w=pw[i]
        if w<3 or w>140: continue
        seg=np.array(path[max(0,i-6):i+7],float)
        if len(seg)>=3:
            c0=seg-seg.mean(0); vt=np.linalg.svd(c0,full_matrices=False)[2]; perp=np.array([-vt[0][1],vt[0][0]])
        else: perp=np.array([0.,1.])
        out.append((w,(y,x),perp,np.array(path)))
    return out
def render(mhc,dapi,cls,zoom_targets):
    disp=((mhc-mhc.min())/(mhc.max()-mhc.min()+1e-6)*255).astype(np.uint8); db=(np.clip(dapi/np.percentile(dapi,99.5),0,1)*255).astype(np.uint8)
    rgb=np.dstack([disp,np.zeros_like(disp),db.astype(np.uint8)])
    im=Image.fromarray(rgb.copy()); d=ImageDraw.Draw(im)
    for w,(y,x),perp,path in cls:
        for yy,xx in path: d.point((xx,yy),fill=(0,220,255))
        dxp,dyp=perp[1]*w*PXUM/2,perp[0]*w*PXUM/2
        d.line([(x-dxp,y-dyp),(x+dxp,y+dyp)],fill=(255,255,0),width=2)
        d.ellipse([x-3,y-3,x+3,y+3],fill=(255,255,255))
        d.text((int(x)+4,int(y)-7),f"{w:.0f}",fill=(150,255,150))
    # zoom crops
    zooms=[]
    for (w,(y,x),perp,path) in zoom_targets:
        R=90; y0,x0=max(0,y-R),max(0,x-R)
        crop=rgb[y0:y0+2*R,x0:x0+2*R].copy(); ci=Image.fromarray(crop).resize((360,360),Image.NEAREST); cd=ImageDraw.Draw(ci)
        sc=360/(2*R); cy,cx=(y-y0)*sc,(x-x0)*sc
        for yy,xx in path:
            if y0<=yy<y0+2*R and x0<=xx<x0+2*R: cd.point(((xx-x0)*sc,(yy-y0)*sc),fill=(0,220,255))
        dxp,dyp=perp[1]*w*PXUM/2*sc,perp[0]*w*PXUM/2*sc
        cd.line([(cx-dxp,cy-dyp),(cx+dxp,cy+dyp)],fill=(255,255,0),width=3)
        cd.ellipse([cx-5,cy-5,cx+5,cy+5],fill=(255,255,255))
        cd.text((6,6),f"{w:.0f} um  (thickest, perpendicular)",fill=(150,255,150)); zooms.append(ci)
    return im,zooms
picks=[("E40 (test)",ROOT+"/E40 hpMb L/scrambled 3d/20x_01"),
       ("E44 (test, unseen lab)",ROOT+"/E44 hpMb G Klara/3d/BSA/20x_23")]
rowimgs=[]
for tag,fld in picks:
    if not os.path.exists(fld+"/Image_CH3.tif"):
        alt=glob.glob(ROOT+"/E44 hpMb G Klara/3d/*/20x_2*"); fld=[a for a in alt if os.path.exists(a+"/Image_CH3.tif")][0]
    mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))[:,:,0]
    dapi=np.array(Image.open(fld+"/Image_CH1.tif").convert("RGB"))[:,:,2]
    cls=centerlines(semseg(mhc))
    cls_sorted=sorted(cls,key=lambda c:-c[0]); targets=[c for c in cls_sorted if c[0]<70][:2]
    full,zooms=render(mhc,dapi,cls,targets)
    ImageDraw.Draw(full).text((8,8),f"{tag}: {len(cls)} fibers, median {np.median([c[0] for c in cls]):.0f}um",fill=(255,255,255))
    rowimgs.append((full,zooms,tag))
    print(f"{tag}: {len(cls)} fibers | median {np.median([c[0] for c in cls]):.1f}um")
W,H=760,570
canvas=Image.new("RGB",(W+380+16,H*2+40),(16,16,16)); dd=ImageDraw.Draw(canvas)
dd.text((10,6),"DIAMETER PLACEMENT ON TEST SET  —  white dot = thickest point · yellow = perpendicular width · cyan = main spine",fill=(230,230,230))
for i,(full,zooms,tag) in enumerate(rowimgs):
    canvas.paste(full.resize((W,H)),(0,i*(H+18)+22))
    for j,z in enumerate(zooms[:2]): canvas.paste(z.resize((188,188)),(W+8+ (j%1)*0, i*(H+18)+22+j*192))
    for j,z in enumerate(zooms[:2]): canvas.paste(z.resize((372,278)),(W+8, i*(H+18)+22+j*284))
canvas.save(OUT+"/diameter_placement_testset.png"); print("saved -> for_katja/diameter_placement_testset.png")
