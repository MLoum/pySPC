import numpy as np
import matplotlib.pyplot as plt


raw_mdf = np.loadtxt("mdf_gaussian_proba.txt")
raw_mdf = np.transpose(raw_mdf)
# raw_mdf /= raw_mdf.max()
space_step = 50
# print(raw_mdf.shape)
r_mdf = np.arange(raw_mdf.shape[1])*space_step
z_mdf = np.arange(raw_mdf.shape[0])*space_step

# I0
print(np.sum(raw_mdf[:][int(raw_mdf.shape[0]/2)]))
print(raw_mdf.max())

# raw_mdf /= raw_mdf.max()
plt.contourf(r_mdf, z_mdf, raw_mdf, levels=40, cmap=plt.cm.jet)
plt.xlabel("r / nm")
plt.ylabel("z / nm")
plt.colorbar()
plt.savefig("mdf_gaussian_1mw_wait500nm_kwPerCmSquare.png", dpi=400)
plt.show()
