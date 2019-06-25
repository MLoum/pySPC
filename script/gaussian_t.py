import numpy as np
import matplotlib.pyplot as plt
import lmfit

raw = np.loadtxt("c:\\intel\\test_gaussien.txt")
raw = np.transpose(raw)

hist , bin_edges = np.histogram(raw, 1000)

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

gmodel = lmfit.Model(gaussian)
params = gmodel.make_params(cen=0, amp=4000, wid=2)
x = bin_edges[:-1]
result = gmodel.fit(hist, params, x=x)
print(result.fit_report())
plt.plot(bin_edges[:-1], hist )
plt.plot(x, result.best_fit, 'r-')
plt.xlabel("value")
plt.xlabel("occurence")
plt.show()
