import numpy as np, os, glob, time, warnings, random
warnings.filterwarnings("ignore")
from PIL import Image
from scipy import ndimage
import torch, torch.nn as nn, torch.nn.functional as F
from skimage.filters import gaussian
Image.MAX_IMAGE_PIXELS=None
dev="cuda"
TS=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/training_set"
MD=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/models"
HOLDOUT=("E40","E44")
def preprocess(g):
    g=g.astype(np.float32); bg=gaussian(g,40); fg=np.clip(g-bg,0,None)
    lo,hi=np.percentile(fg,[1,99.5]); s=np.clip((fg-lo)/(hi-lo+1e-6),0,1)
    return (0.5*s+0.5*(s**0.6)).astype(np.float32)
def soft_erode(x): return -F.max_pool2d(-x,3,1,1)
def soft_dilate(x): return F.max_pool2d(x,3,1,1)
def soft_open(x): return soft_dilate(soft_erode(x))
def soft_skel(x,it=8):
    x1=soft_open(x); s=F.relu(x-x1)
    for _ in range(it):
        x=soft_erode(x); x1=soft_open(x); d=F.relu(x-x1); s=s+F.relu(d-s*d)
    return s
def soft_dice(p,t,m):
    p=p*m; t=t*m; return 1-(2*(p*t).sum()+1)/((p*p).sum()+(t*t).sum()+1)
def cbdice(p,t,w,m):
    p=p*m; t=t*m; sp=soft_skel(p); st=soft_skel(t)
    tp=(sp*t*w).sum()/((sp*w).sum()+1e-6); ts=(st*p*w).sum()/((st*w).sum()+1e-6)
    return 1-2*tp*ts/(tp+ts+1e-6)
def boundary(p,t):
    pb=p-soft_erode(p); tb=t-soft_erode(t)
    return 1-(2*(pb*tb).sum()+1)/(pb.sum()+tb.sum()+1)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,c=40):
        super().__init__();s.e1=cbr(1,c);s.e2=cbr(c,c*2);s.e3=cbr(c*2,c*4);s.e4=cbr(c*4,c*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(c*8,c*4,2,2);s.d3=cbr(c*8,c*4);s.u2=nn.ConvTranspose2d(c*4,c*2,2,2);s.d2=cbr(c*4,c*2)
        s.u1=nn.ConvTranspose2d(c*2,c,2,2);s.d1=cbr(c*2,c);s.out=nn.Conv2d(c,1,1)
    def forward(s,x):
        e1=s.e1(x);e2=s.e2(s.p(e1));e3=s.e3(s.p(e2));e4=s.e4(s.p(e3))
        d3=s.d3(torch.cat([s.u3(e4),e3],1));d2=s.d2(torch.cat([s.u2(d3),e2],1));d1=s.d1(torch.cat([s.u1(d2),e1],1))
        return s.out(d1)
files=[f for f in glob.glob(TS+"/*.npz") if not any(h in os.path.basename(f) for h in HOLDOUT)]
random.seed(1); random.shuffle(files)
data=[]
for f in files:
    d=np.load(f,allow_pickle=True); ip=str(d["img_path"])
    if not os.path.exists(ip): continue
    g=np.array(Image.open(ip).convert("RGB"))[:,:,0]
    if g.shape[1]>=1900: continue
    pre=preprocess(g); fg=(d["labels"]>0).astype(np.float32)
    valid=np.clip(fg+(pre<0.06).astype(np.float32),0,1)
    rad=ndimage.distance_transform_edt(fg).astype(np.float32)
    data.append((pre,fg,valid,rad))
    if len(data)>=600: break
print(f"loaded {len(data)} fields",flush=True)
net=UNet(40).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
NITER=6000; sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,NITER)
def crop(a,sz=256):
    pre,fg,valid,rad=a; H,W=pre.shape; y=random.randint(0,H-sz); x=random.randint(0,W-sz)
    sl=(slice(y,y+sz),slice(x,x+sz)); fl=random.random()<0.5; vf=random.random()<0.5
    def g(z):
        z=z[sl]
        if fl: z=z[:,::-1]
        if vf: z=z[::-1]
        return z.copy()
    return g(pre),g(fg),g(valid),g(rad)
def batch(bs=8):
    P=[];Fg=[];V=[];R=[]
    for _ in range(bs):
        p,f,v,r=crop(random.choice(data));P.append(p);Fg.append(f);V.append(v);R.append(r)
    t=lambda a:torch.from_numpy(np.stack(a))[:,None].float().to(dev)
    return t(P),t(Fg),t(V),t(R)
net.train(); t0=time.time()
for it in range(NITER):
    P,Fg,V,R=batch(8); logit=net(P); prob=torch.sigmoid(logit)
    bce=(F.binary_cross_entropy_with_logits(logit,Fg,reduction='none')*V).sum()/(V.sum()+1e-6)
    loss=bce + 1.5*soft_dice(prob,Fg,V) + 0.6*cbdice(prob,Fg,R,V) + 0.4*boundary(prob*V,Fg*V)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it%1000==0: print(f"  it{it}: loss={loss.item():.3f} [{time.time()-t0:.0f}s]",flush=True)
torch.save(net.state_dict(),MD+"/stage2b_unet.pt")
print(f"DONE {(time.time()-t0)/60:.1f}min | saved stage2b_unet.pt",flush=True)
