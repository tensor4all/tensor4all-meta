import re,sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
src="/Users/hiroshi/projects/tensor4all/faer-benchmark/result/zen5-cpu/faer-kernels/throughput/throughput-qr_factor-v2.md"
blocks=[];cur=None
for l in open(src):
    if l.startswith("| n "): cur={}; blocks.append(cur)
    m=re.match(r"\| (\d+) \| ([\d.]+) .*?\| ([\d.]+) .*?\| ([\d.]+) .*?\| ([\d.]+)",l)
    if m and cur is not None: cur[int(m[1])]=tuple(float(m[i]) for i in range(2,6))
keep=[32,64,96,128,256,512,1024]
plt.rcParams.update({"font.family":"Helvetica","font.size":15})
fig,axs=plt.subplots(1,2,figsize=(8.4,3.3),sharey=True)
for ax,(b,title) in zip(axs,[(blocks[0],"1 thread"),(blocks[2],"8 threads")]):
    ns=[n for n in keep if n in b]
    ax.plot(ns,[b[n][0] for n in ns],"-o",color="#1e40af",lw=2.6,ms=6,label="our fork (t4a-faer)")
    ax.plot(ns,[b[n][1] for n in ns],"--o",color="#94a3b8",lw=1.8,ms=4,label="upstream faer")
    ax.plot(ns,[b[n][2] for n in ns],"-^",color="#111827",lw=1.6,ms=5,label="MKL")
    ax.set_xscale("log",base=2); ax.set_yscale("log")
    tk=[n for n in ns if n!=96]; ax.set_xticks(tk); ax.set_xticklabels([str(n) for n in tk]); ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([48, 80, 96, 112, 192, 384, 768]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(which="minor", length=4)
    ax.set_title(title,fontsize=18,fontweight="bold",loc="left")
    ax.set_xlabel("matrix size n"); ax.grid(alpha=.25); ax.grid(axis="y", which="minor", alpha=.12)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("GFLOP/s (nominal)")
axs[0].legend(frameon=False,loc="upper left")
fig.tight_layout()
fig.savefig(sys.argv[1])
