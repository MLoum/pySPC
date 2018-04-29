import numpy as np
import matplotlib.pyplot as plt

4,
fig = plt.figure(figsize=(5,5), dpi=250)

nbAngle = 10
angleDegree = np.linspace(0, 180, nbAngle)
angle = angleDegree*  np.pi/180.0

lambda_ =  0.632
n = 1.33




q = 4 * np.pi * np.sin(angle/2) / lambda_
qx = q * np.cos(angle/2)
qy = q * np.sin(angle/2)


# for i in range(nbAngle) :
#     plt.scatter(qx[i], qy[i])

plt.scatter(qx, qy)
offsetText = 0.1

for  i in range(nbAngle):
    s = str(int(angleDegree[i]))
    plt.text(qx[i] + offsetText, qy[i] + offsetText, s, fontsize=15)

plt.xlabel("$q_x$ / $\mu m^{-1}$")
plt.ylabel("$q_y$ / $\mu m^{-1}$")

plt.show()
plt.savefig("espaceFourier.png")