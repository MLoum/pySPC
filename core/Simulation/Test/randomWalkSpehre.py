import numpy as np
import matplotlib.pyplot as plt

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return (x,y,z)

# Example IPython code to test the uniformity of the distribution
from pylab import scatter

threetups = []
for _ in xrange(10000):
    threetups.append(random_three_vector())
zipped = zip(*threetups)

plt.scatter(*zipped[0:2])

plt.show()