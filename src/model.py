"""
Myotube morphometry — network + preprocessing (shared by all scripts).

The model is a compact multi-head U-Net that predicts, per pixel:
  - sem : fiber-vs-background probability (semantic segmentation)
  - emb : an 8-D instance embedding (same fiber -> similar; used for merging)
  - ori : local fiber orientation (cos 2theta, sin 2theta)

Two input variants are used in the project:
  inp=1 : MHC channel only          -> checkpoint output/models/emb_net.pt   (default, no nucleus false positives)
  inp=3 : MHC + DAPI + Myogenin     -> checkpoint output/models/mc_net.pt     (better E44 calibration, but can hallucinate fibers in nucleus-dense regions)
"""
import numpy as np
import torch
import torch.nn as nn
from skimage.filters import gaussian

EMB = 8  # embedding dimension


# ----------------------------------------------------------------------------- preprocessing
def prep_mhc(g):
    """MHC channel -> background-attenuated, faint-boosted [0,1] map (model input)."""
    g = g.astype(np.float32)
    bg = gaussian(g, 40)                      # estimate slowly-varying background
    fg = np.clip(g - bg, 0, None)             # subtract it
    lo, hi = np.percentile(fg, [1, 99.5])
    s = np.clip((fg - lo) / (hi - lo + 1e-6), 0, 1)
    return (0.5 * s + 0.5 * (s ** 0.6)).astype(np.float32)   # blend linear + gamma (lifts faint fibers)


def norm(g):
    """Simple 1-99.5 percentile normalisation to [0,1] (for DAPI / Myogenin channels)."""
    g = g.astype(np.float32)
    lo, hi = np.percentile(g, [1, 99.5])
    return np.clip((g - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- network
def _cbr(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
    )


class MHNet(nn.Module):
    def __init__(self, c=40, inp=1):
        super().__init__()
        self.e1 = _cbr(inp, c); self.e2 = _cbr(c, c * 2); self.e3 = _cbr(c * 2, c * 4); self.e4 = _cbr(c * 4, c * 8)
        self.p = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2); self.d3 = _cbr(c * 8, c * 4)
        self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2); self.d2 = _cbr(c * 4, c * 2)
        self.u1 = nn.ConvTranspose2d(c * 2, c, 2, 2);     self.d1 = _cbr(c * 2, c)
        self.sem = nn.Conv2d(c, 1, 1)
        self.emb = nn.Conv2d(c, EMB, 1)
        self.ori = nn.Conv2d(c, 2, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p(e1)); e3 = self.e3(self.p(e2)); e4 = self.e4(self.p(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        f  = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.sem(f), self.emb(f), self.ori(f)


def load_model(checkpoint, inp=1, device="cuda"):
    net = MHNet(40, inp=inp).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()
    return net
