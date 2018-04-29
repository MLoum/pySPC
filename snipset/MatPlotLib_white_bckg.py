import numpy as np
import matplotlib.pyplot as plt

def get_ax_size(ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

data = np.arange(9).reshape((3, 3))
fig = plt.figure(figsize=(8,6), dpi=80)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(data, aspect='normal')
print(get_ax_size(ax))
# (640.0, 480.0)
#plt.savefig('/tmp/test.png', dpi=80)
plt.show()