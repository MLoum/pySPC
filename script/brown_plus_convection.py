import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

raw = np.loadtxt("particle_xyz_conv.txt")
raw = np.transpose(raw)
x = raw[0]
y = raw[1]
z = raw[2]


tick_duration = 50E-9
bin_time = 1E-6

nb_of_tick = x.size
print(nb_of_tick)

nb_of_tick_per_bin = int(bin_time/tick_duration)

nb_of_bin = int(nb_of_tick / nb_of_tick_per_bin)
mean_x = np.zeros(nb_of_bin)
mean_y = np.zeros(nb_of_bin)
mean_z = np.zeros(nb_of_bin)

print(nb_of_bin)
for i in range(int(nb_of_bin - 1)):
    mean_x[i] = np.mean(x[i*nb_of_tick_per_bin:(i+1)*nb_of_tick_per_bin])
    mean_y[i] = np.mean(y[i * nb_of_tick_per_bin:(i + 1) * nb_of_tick_per_bin])
    mean_z[i] = np.mean(z[i * nb_of_tick_per_bin:(i + 1) * nb_of_tick_per_bin])


fig = plt.figure()
ax = plt.axes(projection='3d')

# ax.plot3D(mean_x[1:-2], mean_y[1:-2], mean_z[1:-2], label="mvt brownien 3D")
# ax.plot3D(mean_x[1:-2], mean_y[1:-2], 600, alpha=0.5, label="Projection xy")

color_sequence = np.arange(mean_x[1:-2].size) / mean_x[1:-2].size

ax.scatter(mean_x[1:-2], mean_y[1:-2], mean_z[1:-2], c=color_sequence, cmap=plt.cm.magma, marker='o', alpha=0.7, s=2)
ax.plot3D(mean_x[1:-2], mean_y[1:-2], mean_z[1:-2], label="mvt brownien 3D", alpha=0.1)

ax.scatter(mean_x[1:-2], mean_y[1:-2], 600, c=color_sequence, cmap=plt.cm.magma, marker='o', alpha=0.2, s=2)
ax.plot3D(mean_x[1:-2], mean_y[1:-2], 600, label="Projection xy", alpha=0.1)


ax.scatter(mean_x[1], mean_y[1], mean_z[1], marker='o', c='g')
ax.scatter(mean_x[-2], mean_y[-2], mean_z[-2], marker='^', c='r')
ax.scatter(mean_x[1], mean_y[1], 550, marker='o', c='g', alpha=0.4)
ax.scatter(mean_x[-2], mean_y[-2], 550, marker='^', c='r', alpha=0.4)
ax.set_xlabel('x / nm')
ax.set_ylabel('y / nm')
ax.set_zlabel('z / nm')
plt.legend()
plt.savefig("mvt_brownien_3D_conv.png", dpi=400)
plt.show()