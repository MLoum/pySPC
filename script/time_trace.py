import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use('ggplot')
# plt.rcParams['lines.linewidth'] = 4

raw = np.loadtxt("Burst_time_trace.txt")
raw = np.transpose(raw)

chronoPlotX = raw[0]
chronoPlot = raw[1]
chronoPlotX -= chronoPlotX[0]

plt.plot(chronoPlotX, chronoPlot)
plt.fill_between(chronoPlotX, 0, chronoPlot, alpha=0.3)
plt.xlabel("time /µs")
plt.ylabel("nb of photon per ms")

ax = plt.gca()

ax.add_patch(
    patches.Rectangle(
        (350000, 0),  # (x,y)
        400000,  # width
        chronoPlot.max(),  # height
        alpha=0.2
    )
)

ax.add_patch(
    patches.Rectangle(
        (840000, 0),  # (x,y)
        100000,  # width
        chronoPlot.max(),  # height
        alpha=0.2
    )
)
plt.text(400000, 600, "burst 1")
plt.text(840000, 600, "burst 2")


plt.savefig("time_trace.png", dpi=400)
plt.show()

raw = np.loadtxt("lifetime_burst1.txt")
raw = np.transpose(raw)
microtime = raw[0]
histo = raw[1]
histo /= histo.max()

raw = np.loadtxt("lifetime_burst2.txt")
raw = np.transpose(raw)
microtime_2 = raw[0]
histo_2 = raw[1]
histo_2 /= histo_2.max()

plt.semilogy(microtime, histo, linewidth=0.1, marker="o", markersize=5, label="burst 1")
plt.semilogy(microtime_2, histo_2, linewidth=0.1, marker="^", markersize=5, label="burst 2")
plt.xlim(0,10)
plt.ylim(0.05,1.1)
plt.xlabel("delay time in ns")
plt.ylabel("Normalized intensity")
plt.legend()
plt.savefig("burst_lifetime.png", dpi=400)
plt.show()