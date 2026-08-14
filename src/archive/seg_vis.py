import numpy as np, os, glob, warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, binary_closing, disk, remove_small_holes
Image.MAX_IMAGE_PIXELS=None
dev="cuda"; EMB=8
ROOT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/Katja Myoblasts"
MD=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/models"
OUT=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/for_katja"
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
net=MHNet(40).to(dev); net.load_state_dict(torch.load(MD+"/mc_net.pt")); net.eval()
def seg_mask(mhc,dapi,myog):
    x3=np.stack([prep_mhc(mhc),norm(dapi),norm(myog)]).astype(np.float32)
    with torch.no_grad(): sem,_,_=net(torch.from_numpy(x3)[None].float().to(dev))
    prob=torch.sigmoid(sem)[0,0].cpu().numpy()
    return binary_closing(remove_small_holes(remove_small_objects(prob>0.45,300),2000),disk(4))
# find 2 fields with marked overlays (2 different conditions)
fields=[]
for cond in ["scrambled 3d/20x_01","KIF3_20x_04_r".replace("KIF3","")]:
    pass
cands=[]
for dp,dn,fn in os.walk(ROOT+"/E40 hpMb L"):
    if os.path.exists(dp+"/Image_CH3.tif"):
        m=glob.glob(dp+"/*Overlay m*.tif")+glob.glob(dp+"/*measure*.tif")
        if m: cands.append((dp,m[0]))
cands=sorted(cands)
picks=[cands[0], cands[len(cands)//2]]   # two different fields
panels=[]
for fld,marked in picks:
    mhc=np.array(Image.open(fld+"/Image_CH3.tif").convert("RGB"))[:,:,0]
    dapi=np.array(Image.open(fld+"/Image_CH1.tif").convert("RGB"))[:,:,2].astype(np.float32)
    myog=np.array(Image.open(fld+"/Image_CH2.tif").convert("RGB"))[:,:,1] if os.path.exists(fld+"/Image_CH2.tif") else np.zeros_like(mhc)
    mask=seg_mask(mhc,dapi,myog)
    disp=((mhc-mhc.min())/(mhc.max()-mhc.min()+1e-6)*255).astype(np.uint8); db=(norm(dapi)*255).astype(np.uint8)
    base=np.dstack([disp,np.zeros_like(disp),db])       # red MHC + blue DAPI (like her overlay)
    edge=mask^ndimage.binary_erosion(mask,iterations=2)  # thick outline
    ours=base.copy(); ours[edge]=[0,255,0]              # green = OUR fiber boundary
    ours_im=Image.fromarray(ours); ImageDraw.Draw(ours_im).text((8,8),"OUR fiber segmentation (green outline)",fill=(255,255,255))
    katja=Image.open(marked).convert("RGB").resize((mhc.shape[1],mhc.shape[0]))
    ImageDraw.Draw(katja).text((8,8),"KATJA (calipers + nuclei marks)",fill=(255,255,255))
    name=os.path.relpath(fld,ROOT)
    panels.append((ours_im,katja,name))
W,H=960,720
canvas=Image.new("RGB",(W*2+10,H*2+40),(15,15,15)); d=ImageDraw.Draw(canvas)
for i,(o,kj,name) in enumerate(panels):
    canvas.paste(o.resize((W,H)),(0,i*(H+20)+20)); canvas.paste(kj.resize((W,H)),(W+10,i*(H+20)+20))
    d.text((8,i*(H+20)+4),f"Field {i+1}: {name}",fill=(255,255,0))
canvas.save(OUT+"/segmentation_vs_katja_2fields.png")
print("saved -> for_katja/segmentation_vs_katja_2fields.png")
for _,_,n in panels: print("  field:",n)
