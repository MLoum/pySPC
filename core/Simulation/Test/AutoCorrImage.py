import numpy as np
import matplotlib.pyplot as plt

#TODO Normalisation

def autocorr(im):
    dimX, dimY = np.shape(im)
    result = np.zeros(dimX)

    #meanSquare = np.mean(x)**2 #* length/np.size(x)
    result[0] = np.sum(im**2)
    for j in range(1, dimX):
        print(j)
        #The default (axis = None) is perform a sum over all the dimensions of the input array.
        result[j] = np.sum(im[0:-j, :] * im[j:, :])
    #result /= np.size(x)
    #result /= meanSquare

    # normCoeff = np.mean(x ** 2) - np.mean(x) * 2
    # result /= normCoeff
    return result

nbPoint = 100

im = np.random.rand(nbPoint, nbPoint)

# I= np.zeros(nbPoint)
# I[0:500] = 5

#plt.plot(I)
plt.plot(autocorr(im))
plt.show()
