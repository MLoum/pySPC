import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

limite_mu = 15

x = np.linspace(0,limite_mu,100000)
mu = 49

poisson = np.exp(-mu)/ factorial(x) * np.power(mu, x)
# plt.plot(x, poisson)
# plt.show()


mus = np.linspace(0,limite_mu,1000)
mode = np.zeros(mus.size)

i = 0
for mu in mus:
    poisson = np.exp(-mu) / factorial(x) * np.power(mu, x)
    max = x[np.argmax(poisson)]
    mode[i] = max
    i += 1
    print(mu)

plt.plot(mus, mode - mus)
# plt.plot(x,x, "k--")
plt.show()


