import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv


NA = 0.75
lambda_nm = 405
airy_radius_nm = 1.22 * lambda_nm / NA

x = np.linspace(-3,3,300)

airy = np.power(2*jv(1, x*np.pi) / (x*np.pi), 2)



plt.plot(x*airy_radius_nm, airy, label="Airy")
plt.xlabel("focal plane distance (nm)")
plt.ylabel("Normalized intensity")

plt.show()