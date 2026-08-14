import numpy as np, os, glob, time, warnings, random
warnings.filterwarnings("ignore")
from PIL import Image
from scipy import ndimage
from skimage.feature import structure_tensor
import torch, torch.nn as nn, torch.nn.functional as F
from skimage.filters import gaussian
Image.MAX_IMAGE_PIXELS=None
dev="cuda"
TS=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/training_set"
MD=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/models"
HOLD=("E40","E44"); EMB=8
def preprocess(g):
    g=g.astype(np.float32); bg=gaussian(g,40); fg=np.clip(g-bg,0,None)
    lo,hi=np.percentile(fg,[1,99.5]); s=np.clip((fg-lo)/(hi-lo+1e-6),0,1)
    return (0.5*s+0.5*(s**0.6)).astype(np.float32)
def orient(img):
    Arr,Arc,Acc=structure_tensor(img.astype(np.float32),sigma=3,order='rc')
    th=0.5*np.arctan2(2*Arc,(Arr-Acc)+1e-6)
    return np.stack([np.cos(2*th),np.sin(2*th)],0).astype(np.float32)
def soft_erode(x): return -F.max_pool2d(-x,3,1,1)
def soft_dilate(x): return F.max_pool2d(x,3,1,1)
def soft_open(x): return soft_dilate(soft_erode(x))
def soft_skel(x,it=8):
    x1=soft_open(x); s=F.relu(x-x1)
    for _ in range(it):
        x=soft_erode(x); x1=soft_open(x); d=F.relu(x-x1); s=s+F.relu(d-s*d)
    return s
def soft_dice(p,t,m): p=p*m;t=t*m; return 1-(2*(p*t).sum()+1)/((p*p).sum()+(t*t).sum()+1)
def cbdice(p,t,w,m):
    p=p*m;t=t*m;sp=soft_skel(p);st=soft_skel(t)
    tp=(sp*t*w).sum()/((sp*w).sum()+1e-6);ts=(st*p*w).sum()/((st*w).sum()+1e-6)
    return 1-2*tp*ts/(tp+ts+1e-6)
def disc_loss(emb,inst,dv=0.5,dd=1.5):
    tot=emb.sum()*0; nb=0
    for b in range(emb.shape[0]):
        e=emb[b]; lab=inst[b,0].long(); ids=torch.unique(lab); ids=ids[ids>0]
        if len(ids)<1: continue
        means=[]; lv=0.
        for i in ids:
            m=(lab==i)
            if m.sum()<5: continue
            ei=e[:,m]; mu=ei.mean(1); means.append(mu)
            lv=lv+F.relu(torch.norm(ei-mu[:,None],dim=0)-dv).pow(2).mean()
        if len(means)<1: continue
        means=torch.stack(means,0); lreg=torch.norm(means,dim=1).mean()
        ld=torch.tensor(0.,device=dev)
        if len(means)>1:
            dm=torch.cdist(means,means); K=len(means); msk=~torch.eye(K,dtype=bool,device=dev)
            ld=F.relu(2*dd-dm[msk]).pow(2).mean()
        tot=tot+lv/len(means)+ld+0.001*lreg; nb+=1
    return tot/max(nb,1)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
class MHNet(nn.Module):
    def __init__(s,c=40):
        super().__init__();s.e1=cbr(1,c);s.e2=cbr(c,c*2);s.e3=cbr(c*2,c*4);s.e4=cbr(c*4,c*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(c*8,c*4,2,2);s.d3=cbr(c*8,c*4);s.u2=nn.ConvTranspose2d(c*4,c*2,2,2);s.d2=cbr(c*4,c*2)
        s.u1=nn.ConvTranspose2d(c*2,c,2,2);s.d1=cbr(c*2,c)
        s.sem=nn.Conv2d(c,1,1); s.emb=nn.Conv2d(c,EMB,1); s.ori=nn.Conv2d(c,2,1)
    def forward(s,x):
        e1=s.e1(x);e2=s.e2(s.p(e1));e3=s.e3(s.p(e2));e4=s.e4(s.p(e3))
        d3=s.d3(torch.cat([s.u3(e4),e3],1));d2=s.d2(torch.cat([s.u2(d3),e2],1));f=s.d1(torch.cat([s.u1(d2),e1],1))
        return s.sem(f),s.emb(f),s.ori(f)
files=[f for f in glob.glob(TS+"/*.npz") if not any(h in os.path.basename(f) for h in HOLD)]
random.seed(2); random.shuffle(files); data=[]
for f in files:
    d=np.load(f,allow_pickle=True); ip=str(d["img_path"])
    if not os.path.exists(ip): continue
    g=np.array(Image.open(ip).convert("RGB"))[:,:,0]
    if g.shape[1]>=1900: continue
    pre=preprocess(g); fgm=(d["labels"]>0).astype(np.float32)
    valid=np.clip(fgm+(pre<0.06).astype(np.float32),0,1)
    rad=ndimage.distance_transform_edt(fgm).astype(np.float32)
    data.append((pre,fgm,valid,rad,d["labels"].astype(np.int32)))
    if len(data)>=350: break
print(f"loaded {len(data)} fields",flush=True)
net=MHNet(40).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
NITER=int(os.environ.get("NITER","6000")); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,NITER)
def crop(a,sz=256):
    pre,fg,valid,rad,inst=a; H,W=pre.shape; y=random.randint(0,H-sz); x=random.randint(0,W-sz)
    sl=(slice(y,y+sz),slice(x,x+sz))
    ori=orient(pre[sl])
    return pre[sl].copy(),fg[sl].copy(),valid[sl].copy(),rad[sl].copy(),inst[sl].copy(),ori
def batch(bs=6):
    P=[];Fg=[];V=[];R=[];I=[];O=[]
    for _ in range(bs):
        p,f,v,r,i,o=crop(random.choice(data));P.append(p);Fg.append(f);V.append(v);R.append(r);I.append(i);O.append(o)
    t=lambda a:torch.from_numpy(np.stack(a))[:,None].float().to(dev)
    return t(P),t(Fg),t(V),t(R),t(I),torch.from_numpy(np.stack(O)).float().to(dev)
net.train(); t0=time.time()
for it in range(NITER):
    P,Fg,V,R,I,O=batch(6); sem,emb,ori=net(P); prob=torch.sigmoid(sem)
    bce=(F.binary_cross_entropy_with_logits(sem,Fg,reduction='none')*V).sum()/(V.sum()+1e-6)
    Lsem=bce+1.5*soft_dice(prob,Fg,V)+0.6*cbdice(prob,Fg,R,V)
    Ldisc=disc_loss(emb,I)
    on=F.normalize(ori,dim=1); Lori=(((on-O)**2).sum(1,keepdim=True)*Fg).sum()/(Fg.sum()+1e-6)
    loss=Lsem+1.0*Ldisc+0.3*Lori
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it%(2 if NITER<20 else 1000)==0: print(f"  it{it}: L={loss.item():.3f} (sem {Lsem.item():.2f} disc {Ldisc.item():.2f} ori {Lori.item():.2f}) [{time.time()-t0:.0f}s]",flush=True)
if NITER>=20:
    torch.save(net.state_dict(),MD+"/emb_net.pt"); print(f"DONE {(time.time()-t0)/60:.1f}min | saved emb_net.pt",flush=True)
else: print("SMOKE OK",flush=True)
