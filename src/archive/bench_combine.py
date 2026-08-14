import csv, numpy as np, os
from scipy.stats import pearsonr
out=r"C:/personal/ummunflueres/Maltzahn Examples/Maltzahn Examples/output/benchmark"
def load(f):
    d={}
    for r in csv.DictReader(open(f)):
        k=(r["cond"],r["field"])
        d[k]=dict(treat=r["treat"],tp=r["tp"],n=int(r["n_inst"]),
                  auto=float(r["auto_mean"]),katja=float(r["katja_mean"]))
    return d
cp=load(out+"/e40_cellposesam.csv"); om=load(out+"/e40_omnipose.csv")
keys=sorted(set(cp)&set(om))
print(f"fields compared (both models): {len(keys)}\n")
def summ(name,d,keys):
    a=np.array([d[k]["auto"] for k in keys]); g=np.array([d[k]["katja"] for k in keys])
    n=np.array([d[k]["n"] for k in keys])
    r=pearsonr(a,g)[0]
    print(f"=== {name} ===")
    print(f"  per-field diam: r={r:.3f} | bias={np.mean(a-g):+.1f}um | ratio={np.mean(a/g):.2f} | within5um={100*np.mean(np.abs(a-g)<=5):.0f}%")
    print(f"  mean auto={a.mean():.1f} vs katja={g.mean():.1f} | mean #instances/field={n.mean():.0f} (katja~17)")
    for tp in ["3d","5d"]:
        kk=[k for k in keys if d[k]["tp"]==tp]
        aa=np.array([d[k]["auto"] for k in kk]); gg=np.array([d[k]["katja"] for k in kk])
        print(f"    {tp}: r={pearsonr(aa,gg)[0]:.2f} bias={np.mean(aa-gg):+.1f} ratio={np.mean(aa/gg):.2f}")
    return a,g,r
acp,g,rcp=summ("Cellpose-SAM",cp,keys); print()
aom,_,rom=summ("Omnipose (cyto2)",om,keys)
print(f"\nHEAD-TO-HEAD (which is closer to expert per field): ",
      f"Cellpose-SAM better in {100*np.mean(np.abs(acp-g)<np.abs(aom-g)):.0f}% of fields")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(13,5.5))
for a,name,c,r in [(acp,"Cellpose-SAM","#1f77b4",rcp),(aom,"Omnipose","#ff7f0e",rom)]:
    ax[0].scatter(g,a,s=22,alpha=.6,c=c,label=f"{name} (r={r:.2f}, bias {np.mean(a-g):+.0f}µm)")
lim=[0,max(g.max(),acp.max(),aom.max())+3]; ax[0].plot(lim,lim,'k--',lw=1); ax[0].set_xlim(lim);ax[0].set_ylim(lim)
ax[0].set_xlabel("Expert (Katja) mean diameter µm");ax[0].set_ylabel("Model mean diameter µm")
ax[0].set_title("SOTA baselines vs expert — E40 (58 fields)");ax[0].legend()
# per-condition bias bars
import collections
treats=sorted(set(cp[k]["treat"] for k in keys))
x=np.arange(len(treats)); w=0.35
for i,(d,name,c) in enumerate([(cp,"Cellpose-SAM","#1f77b4"),(om,"Omnipose","#ff7f0e")]):
    bias=[np.mean([d[k]["auto"]-d[k]["katja"] for k in keys if d[k]["treat"]==t]) for t in treats]
    ax[1].bar(x+(i-0.5)*w,bias,w,label=name,color=c)
ax[1].axhline(0,color='k',lw=1);ax[1].set_xticks(x);ax[1].set_xticklabels(treats,rotation=45,ha='right')
ax[1].set_ylabel("mean bias (auto-expert) µm");ax[1].set_title("Per-condition bias");ax[1].legend()
fig.tight_layout(); fig.savefig(out+"/benchmark_cpsam_vs_omni.png",dpi=120)
print("\nsaved -> benchmark_cpsam_vs_omni.png")
