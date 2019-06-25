import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1E-7, 0.01, 1000)
poisson_1 = mu * np.exp(-mu)

relative_diff = np.abs(mu - poisson_1)
print(relative_diff)

plt.plot(mu, relative_diff)
# plt.plot(mu, mu)

plt.show()