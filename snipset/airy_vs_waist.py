import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv


x = np.linspace(-2*2*np.pi,2*2*np.pi,500)

airy = np.power(2*jv(1, x) / (x), 2)

waist = 2.58
sigma = waist/2
# Attention  on trouve plutôt le np.exp(-x**2/(2*sigma**2))
# gaussian =  1/(sigma*np.sqrt(2*np.pi)) * np.exp(-x**2/(sigma * waist**2))
# Intensity is gaussian SQUARED
gaussian_intensity = 1 * np.exp(-2*x**2/((waist)**2))

alpha_line = 0.3
alpha_fill = 0.2
plt.title("Comparaison Tache Airy et Gaussienne")
plt.plot(x/(2*np.pi), airy, label="Airy")
plt.fill_between(x/(2*np.pi), airy, 0.5, where=airy>0.5, alpha=alpha_fill)
plt.plot(x/(2*np.pi), gaussian_intensity, label="Gaussian")
plt.fill_between(x/(2*np.pi), airy, 0.13, where=airy>0.13, alpha=alpha_fill)
plt.fill_between(x/(2*np.pi), airy, 0, where=airy>0, alpha=alpha_fill)
plt.vlines([-1.61/(2*np.pi), 1.61/(2*np.pi)], ymin=0, ymax=1, alpha=alpha_line, linestyle="-.", label="FWHM 50%")
plt.vlines([-2.58/(2*np.pi), 2.58/(2*np.pi)], ymin=0, ymax=1, alpha=alpha_line, linestyle="--", label="waist 13%")
plt.vlines([-3.83/(2*np.pi), 3.8/(2*np.pi)], ymin=0, ymax=1, alpha=alpha_line, label="Airy 0%")
plt.hlines(0.5, xmin=-5, xmax=5, alpha=alpha_line, linestyle="-.")
plt.hlines(0.13, xmin=-5, xmax=5, alpha=alpha_line, linestyle="--")
plt.xlabel("focal plane distance (nm)")
plt.ylabel("Normalized intensity")
plt.xlim(-1,1)
plt.legend()
plt.savefig("airy_vs_gaussian.png", dpi=300)
plt.show()
