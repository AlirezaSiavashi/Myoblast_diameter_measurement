"""
Gap-tolerant Dynamic Snake Convolution (GS-Conv) for myotube fibers.

Background. Dynamic Snake Convolution (DSCNet, ICCV'23) adapts a kernel to crawl ALONG tubular
structures (vessels, roads). It assumes the structure is a *continuous ridge of bright pixels*:
each kernel tap steps 1 px along an axis and shifts <=1 px perpendicular, following the signal.

Why vanilla DSConv does not transfer here. Immunofluorescent myotubes have SIGNAL DROPOUTS -- dim
patches where MHC staining is weak but the fiber physically continues. A snake that only follows
bright pixels either stalls at the gap or veers into a neighbouring (crossing) fiber, which is
exactly the failure mode we are trying to remove.

Our two adaptations:
  (1) DIRECTIONAL MOMENTUM. The perpendicular offset is integrated with inertia,
          dir_c = m * dir_{c-1} + (1 - m) * tanh(delta_c),     pos_c = pos_{c-1} + dir_c
      so across a dim gap (where delta ~ 0 because there is no evidence) the kernel keeps going
      STRAIGHT along the fiber's current direction instead of curling toward brighter neighbours.
      This encodes the prior "a fiber continues through a gap".
  (2) DILATED SNAKE STEPS. Taps are spaced `dil` px apart along the axis, so a k-tap snake spans
      k*dil px in one layer (e.g. 9 taps x dil 4 = 33 px) -- a large *anisotropic* receptive field
      that grows along the fiber, which is where long-range context is actually needed.

Both x-axis and y-axis snakes are used, so arbitrarily oriented fibers are covered.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GapSnakeConv(nn.Module):
    """Gap-tolerant dynamic snake convolution (one axis). Residual bottleneck form: cin -> cmid -> cin."""

    def __init__(self, ch, k=9, dil=4, momentum=0.7, axis=0, reduce=2):
        super().__init__()
        assert k % 2 == 1
        self.k, self.dil, self.m, self.axis = k, dil, momentum, axis
        cm = max(8, ch // reduce)
        self.inc = nn.Conv2d(ch, cm, 1)
        self.off = nn.Conv2d(cm, k, 3, padding=1)      # per-tap perpendicular delta
        self.agg = nn.Conv2d(cm * k, ch, 1)
        self.bn = nn.BatchNorm2d(ch)
        nn.init.zeros_(self.off.weight); nn.init.zeros_(self.off.bias)   # start as a straight dilated kernel
        nn.init.zeros_(self.agg.weight); nn.init.zeros_(self.agg.bias)   # block starts as EXACT identity
        nn.init.zeros_(self.bn.weight); nn.init.zeros_(self.bn.bias)     # -> can warm-start from a trained U-Net

    def _positions(self, delta):
        """delta: (B,k,H,W) raw. Returns perpendicular offsets per tap with directional momentum."""
        B, k, H, W = delta.shape
        N = k // 2
        d = torch.tanh(delta)
        pos = [None] * k
        pos[N] = torch.zeros_like(d[:, 0])
        # outward from the centre, integrating direction with inertia
        dirn = torch.zeros_like(d[:, 0])
        for c in range(1, N + 1):                       # forward (+)
            dirn = self.m * dirn + (1 - self.m) * d[:, N + c]
            pos[N + c] = pos[N + c - 1] + dirn
        dirn = torch.zeros_like(d[:, 0])
        for c in range(1, N + 1):                       # backward (-)
            dirn = self.m * dirn + (1 - self.m) * d[:, N - c]
            pos[N - c] = pos[N - c + 1] - dirn
        return pos

    def forward(self, x):
        B, C, H, W = x.shape
        f = self.inc(x)
        pos = self._positions(self.off(f))
        N = self.k // 2
        dev = x.device
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        outs = []
        for c in range(-N, N + 1):
            along = c * self.dil
            perp = pos[c + N] * self.dil                 # scale perpendicular travel with dilation
            if self.axis == 0:                           # snake along x, bends in y
                sx = xx[None] + along
                sy = yy[None] + perp
            else:                                        # snake along y, bends in x
                sx = xx[None] + perp
                sy = yy[None] + along
            gx = (2.0 * sx / max(W - 1, 1) - 1.0).expand(B, H, W)
            gy = (2.0 * sy / max(H - 1, 1) - 1.0).expand(B, H, W)
            grid = torch.stack([gx, gy], dim=-1)         # (B,H,W,2)
            outs.append(F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=True))
        return F.relu(x + self.bn(self.agg(torch.cat(outs, 1))))


class SnakeBlock(nn.Module):
    """x-axis + y-axis gap-tolerant snakes, so fibers of any orientation get long-range context."""

    def __init__(self, ch, k=9, dil=4, momentum=0.7):
        super().__init__()
        self.sx = GapSnakeConv(ch, k=k, dil=dil, momentum=momentum, axis=0)
        self.sy = GapSnakeConv(ch, k=k, dil=dil, momentum=momentum, axis=1)

    def forward(self, x):
        return self.sy(self.sx(x))
