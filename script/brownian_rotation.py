import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D





raw = np.loadtxt("particle_rotation.txt")
raw = np.transpose(raw)
x = raw[0]
y = raw[1]
z = raw[2]

tick_time = 50E-9
rotation_time = 0.1E-6

nb_point = int(1000*rotation_time/tick_time)
print(nb_point)
x = x[0:nb_point]
y = y[0:nb_point]
z = z[0:nb_point]

# norm = np.sqrt(x**2 + y**2 + z**2)
# plt.plot(norm)
# plt.show()

# tick_duration = 5E-9
# bin_time = 1E-6
#
# nb_of_tick = x.size
# print(nb_of_tick)
#
# nb_of_tick_per_bin = int(bin_time/tick_duration)
#
# nb_of_bin = int(nb_of_tick / nb_of_tick_per_bin)
# mean_x = np.zeros(nb_of_bin)
# mean_y = np.zeros(nb_of_bin)
# mean_z = np.zeros(nb_of_bin)
#
# print(nb_of_bin)
# for i in range(nb_of_bin-1):
#     mean_x[i] = np.mean(x[i*nb_of_tick_per_bin:(i+1)*nb_of_tick_per_bin])
#     mean_y[i] = np.mean(y[i * nb_of_tick_per_bin:(i + 1) * nb_of_tick_per_bin])
#     mean_z[i] = np.mean(z[i * nb_of_tick_per_bin:(i + 1) * nb_of_tick_per_bin])


fig = plt.figure()
ax = plt.axes(projection='3d')

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xb = np.cos(u)*np.sin(v)
yb = np.sin(u)*np.sin(v)
zb = np.cos(v)
ax.plot_surface(xb, yb, zb, color="b", alpha=0.1)
ax.set_aspect("equal")


# ax.plot3D(mean_x[1:-2], mean_y[1:-2], mean_z[1:-2], label="mvt brownien 3D")
# ax.plot3D(mean_x[1:-2], mean_y[1:-2], 600, alpha=0.5, label="Projection xy")

color_sequence = np.arange(x[1:-2].size) / x[1:-2].size

ax.scatter(x[1:-2], y[1:-2], z[1:-2], c=color_sequence, cmap=plt.cm.magma, marker='o', alpha=0.7, s=2)
ax.plot3D(x[1:-2], y[1:-2], z[1:-2], label="mvt brownien 3D", alpha=0.2)
ax.scatter(x[1], y[1], z[1], marker='o', c='g')
ax.scatter(x[-2], y[-2], z[-2], marker='^', c='r')
ax.set_xlabel('vx')
ax.set_ylabel('vy')
ax.set_zlabel('vz')
# plt.legend()
plt.savefig("mvt_brownien_rotation_100micros_r4.png", dpi=400)
plt.show()