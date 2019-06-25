import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import LinearModel, LorentzianModel

raw = np.loadtxt("c:\\intel\\msd.txt")
raw = np.transpose(raw)

diff_tot = np.diff(raw)
print("mean :", np.mean(diff_tot))

diff = np.diff(raw[0:10000])

md = np.cumsum(diff)
msd = diff**2
msd = np.cumsum(msd)

delta_t = 50
time = np.arange(0, msd.size)*50

D = 0.042923566878981

slop = 2 * D * delta_t

msd_fit = LinearModel()
params = msd_fit.make_params(slope=0, intercept=0)
result = msd_fit.fit(msd, params, x=time)
print(result.fit_report())

plt.plot(time, msd, label=r"$<x^2(t)>$")
plt.plot(time, result.best_fit, 'r-', label="ajustement linéaire", alpha=0.5)
plt.xlabel("temps en ns")
plt.ylabel(r"Déplacement quadratique moyen $<x^2(t)>$")
plt.legend()
plt.savefig("msd.png")
plt.show()

plt.plot(time, md, label=r"$<x(t)>$")
plt.xlabel("temps en ns")
plt.ylabel(r"Déplacement moyen $<x(t)>$")
plt.savefig("md.png")
plt.show()
