import numpy as np
import matplotlib.pyplot as plt


raw_vm = np.loadtxt("velocity_map.txt")
raw_vm = np.transpose(raw_vm)
# print(raw_vm)
# raw_mdf /= raw_mdf.max()
space_step = 50
# print(raw_mdf.shape)
y_mdf = np.arange(raw_vm.shape[1])*space_step/1000.0
z_mdf = np.arange(raw_vm.shape[0])*space_step/1000.0

print(np.sum(raw_vm[:][int(raw_vm.shape[0]/2)]))

# raw_vm /= raw_vm.max()
plt.contourf(y_mdf, z_mdf, raw_vm, levels=40, cmap=plt.cm.jet)
plt.xlabel("y / µm")
plt.ylabel("z / µm")
plt.colorbar()
plt.savefig("velocity_map_w100_h50.png", dpi=400)
plt.show()
