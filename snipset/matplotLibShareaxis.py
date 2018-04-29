import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
#The most basic method of creating an axes is to use the plt.axes function. As we've seen previously,
# by default this creates a standard axes object that fills the entire figure. plt.axes also takes an optional
# argument that is a list of four numbers in the figure coordinate system.
# These numbers represent [left, bottom, width, height] in the figure coordinate system,
# which ranges from 0 at the bottom left of the figure to 1 at the top right of the figure.

ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.6],
                   xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2],
                   ylim=(-1.2, 1.2))

x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x));

plt.show()