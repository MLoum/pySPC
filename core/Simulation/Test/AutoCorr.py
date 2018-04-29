import numpy as np
import matplotlib.pyplot as plt


def autocorr(x, length):
    result = np.zeros(length)
    meanSquare = np.mean(x)**2 #* length/np.size(x)
    for j in range(1, length):
        print(j)
        result[j] = np.sum(x[0:-j] * x[j:]) #- meanSquare

    result /= np.size(x)
    result /= meanSquare

    # normCoeff = np.mean(x ** 2) - np.mean(x) * 2
    # result /= normCoeff
    return result

nbPoint = 1E5

I = np.random.rand(nbPoint)

# I= np.zeros(nbPoint)
# I[0:500] = 5

#plt.plot(I)
plt.plot(autocorr(I, 1000))
plt.show()
