import numpy as np
import matplotlib.pyplot as plt
import matplotlib as m


raw = np.loadtxt("particle_rotation_abs.txt")
raw = np.transpose(raw)
r = raw[0]
z = raw[1]
e_dot_mu_square = raw[2]

tick_duration = 5E-9
bin_time = 1E-6

nb_of_tick = r.size
print(nb_of_tick)

nb_of_tick_per_bin = int(bin_time/tick_duration)

nb_of_bin = int(nb_of_tick / nb_of_tick_per_bin)
mean_r = np.zeros(nb_of_bin)
mean_z = np.zeros(nb_of_bin)
mean_edot_mu = np.zeros(nb_of_bin)

print(nb_of_bin)
for i in range(nb_of_bin-1):
    mean_r[i] = np.mean(r[i*nb_of_tick_per_bin:(i+1)*nb_of_tick_per_bin])
    mean_z[i] = np.mean(z[i * nb_of_tick_per_bin:(i + 1) * nb_of_tick_per_bin])
    mean_edot_mu[i] = np.mean(e_dot_mu_square[i * nb_of_tick_per_bin:(i + 1) * nb_of_tick_per_bin])

f, (ax_contour, ax_counts) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(hspace=0.3)


raw_mdf = np.loadtxt("mdf_gaussian.txt")
raw_mdf = np.transpose(raw_mdf)
print(raw_mdf.max())
raw_mdf /= raw_mdf.max()
space_step = 25
print(raw_mdf.shape)
r_mdf = np.arange(raw_mdf.shape[1])*space_step
z_mdf = np.arange(raw_mdf.shape[0])*space_step


ax_contour.contourf(r_mdf, z_mdf, raw_mdf, levels=10, cmap=plt.cm.binary, vmin=0, vmax=0.5, alpha=0.2)
ax_contour.set_xlabel("r / nm")
ax_contour.set_ylabel("z / nm")


# ax_contour.plot(mean_r[1:-2], mean_z[1:-2], color='blue', marker='o', linewidth=0, markersize=2)

color_sequence = mean_edot_mu/np.max(mean_edot_mu)

ax_contour.scatter(mean_r[1:-2], mean_z[1:-2], c=color_sequence[1:-2], cmap=plt.cm.magma, marker='o', alpha=0.7, s=2)
ax_contour.plot(mean_r[1:-2], mean_z[1:-2], linewidth=1, alpha=0.2)

# Mark the start and end points.
ax_contour.plot(mean_r[1], mean_z[1], 'go')
ax_contour.plot(mean_r[-2], mean_z[-2], 'r^')
# ax_contour.colorbar()
# ax_contour.show()

def load_ptn(filename):

    mainClock_MHz = 20
    timestamps_unit = 1/mainClock_MHz * 1e-6


    """
    Bit:     40-0      40-56     56-64
    Meaning: Timestamp Microtime Channel
    """

    f = open(filename, 'rb')

    # np.fromfile(f, dtype='u4', count=1)[0]

    # ...and then the remaining records containing the photon data
    # ttt_dtype = np.dtype([('timeStamp', '<u2'), ('ch1', '<u2'), ('ch2', '<u2'), ('ch3', '<u2'), ('ch4', '<u2'), ('overflow', '<u2')])
    data = np.fromfile(f, dtype=np.uint64)

    # 0x000FFFFF is 0000 0000 1111  1111 1111 1111 1111
    timestamps = np.bitwise_and(data, 0x000FFFFF).astype(dtype='uint64')
    # 0x0FF00000
    nanotime = np.bitwise_and(data, 0x0FF00000).astype(dtype='uint16')
    detector = np.bitwise_and(data, 0xF0000000).astype(dtype='uint8')
    meta = ""

    return timestamps

timestamps = load_ptn("photon_list.ptn")

num_bin = (timestamps / nb_of_tick_per_bin).astype(np.int64)
data = np.bincount(num_bin)

time_axis = np.arange(0, nb_of_bin, dtype=np.float64)
time_axis *= nb_of_tick_per_bin
time_axis *= 50E-3

# if time_axis.size > data.size:
#     np.pad(data, (0, time_axis.size - data.size), mode = 'constant', constant_values = 0)

data_pad = np.zeros(time_axis.size)
data_pad[0:data.size] = data


ax_counts.plot(time_axis, data_pad)



ax_counts.set_xlabel("temps / µs")
ax_counts.set_ylabel("Nbre Photon")
# ax_counts.show()
f.savefig("brown_MDF_rotation.png", dpi=400)
f.show()

print("OK")