import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['lines.linewidth'] = 4

wl = np.arange(0.4,0.9,0.001)
A = 1.3479
C = 0.089
L = 0.05

alpha = C + A/(wl**4)


plt.plot(wl, alpha)
plt.xlabel("wavelength µm")
plt.ylabel("Attenuation en dB/km")
plt.title("Attenuation of GI50 (Sedi fibre\n extrapolated from constructor data")
plt.savefig("Attenuation_GI50.png", dpi=300)

plt.show()

transmission = np.power(10, -alpha * L/10)
plt.plot(wl, transmission)
plt.xlabel("wavelength µm")
plt.ylabel("Transmission")
plt.title("Transmission of GI50 L=100m")
plt.savefig("transmission_GI50.png", dpi=300)
plt.show()
