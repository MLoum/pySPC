import numpy as np
import matplotlib.pyplot as plt


omega_1 = 820
omega_2 = 1800
omega = np.linspace(-20000,20000,500)

omega /= np.pi

spectre1 = omega_1/(omega_1**2 + omega**2)/0.0012
spectre2 = omega_2/(omega_2**2 + omega**2)/0.0004/1.36

plt.plot(omega, spectre1, label=r'$\theta$=30')
plt.plot(omega, spectre2, '--', label=r'$\theta$=45')
plt.xlabel("Freq en Hz")
plt.ylabel("Intensité $I_2(\omega, q)$ (norm.)")
plt.legend()
plt.show()