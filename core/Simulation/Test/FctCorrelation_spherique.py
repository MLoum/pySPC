import numpy as np
import matplotlib.pyplot as plt


tau_1 = 560
tau_2 = 330
t = np.linspace(0,3000,500)

declin_1 = np.exp(-t/tau_1) + 1
declin_2 = np.exp(-t/tau_2) + 1

plt.plot(t, declin_1, label=r'$\theta$=30')
plt.plot(t, declin_2, '--', label=r'$\theta$=45')
plt.xlabel("Temps en $\mu$s")
plt.ylabel("Autocorrélation (norm)")
plt.legend()
plt.show()