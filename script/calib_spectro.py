import numpy as np
import  matplotlib.pyplot as plt

e = 10.0

peaks = []

p = 1
while e/p > 0.35:
    if e/p < 1:
        peaks.append(e/p)
    p += 1

print("nbre pics  dans le visible : ", len(peaks))
print(peaks)
plt.scatter(np.arange(len(peaks)), peaks)
plt.show()

delta_lambda = np.diff(peaks)
print("nbre pics  dans le visible : ", len(peaks))
print(peaks)
plt.scatter(peaks[0:-2]*1000, delta_lambda*1000)
plt.show()

# A partir du deltaPeak, peut-on remonter à e puis aux lambdas d'extinction.