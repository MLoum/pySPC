# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:29:33 2021

@author: MatthieuL
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Calcul de la Molecular Detection Efficieny en 3D avec modèle vectoriel de base pour la PSF d'excitation
et le modèle géométrique de Qian et Elson pour le filtrage confocal.

"""

# On veut calculer le profil selon z en x=y=0
# On veut calculer le profil selon x (ou y) en z = 0


# basic parameters
pi = np.pi
NA = 0.75  # numerical aperture of objective lens
G = 20
f = 180 / G  # en mm
n = 1  # refractive index of immersion medium
lambda_ = 375.e-9  # wavelength of light
alpha = np.arcsin(NA / n)  # maximum open angle of OL
k = 2 * pi * n / lambda_  # wavenumber
lens_half_diameter_mm = f * np.tan(alpha)
print("lens_half_diameter_mm : ", lens_half_diameter_mm)

# pupill filling
w_centre = 0.8  # en mm, diameter
percentage_filling = w_centre / 2 / lens_half_diameter_mm * 100
print("percentage_filling (%): ", percentage_filling)

# confocal filtering
lambda_em_nm = 500
lambda_em = lambda_em_nm * 1E-9
theta_ = np.arcsin(NA / n)
# A la collection on prend TOUTE l'ouverture numérique
# theta_ = np.arctan((w_centre/2)/lens_half_diameter_mm)
print("theta ° : ", theta_ * 180 / np.pi)
print("tan(theta) : ", np.tan(theta_))
print("effective NA ; ", n * np.sin(theta_))
print("lambda_em ", lambda_em)
w = lambda_em / (np.pi * np.tan(theta_))
print("w :", w)

exploration_factor_z = 6
threshold_MDF_detection = 0.2

diam_confocal_pinhole_microns = 25.0
diam_confocal_pinhole = diam_confocal_pinhole_microns * 1E-6
a = diam_confocal_pinhole / G / 2	# radius
print("a :", a)

# image plane in Cartesian coordinates
L_focal = 2.0e-6;  # observation scale
Nx = 51  # discretization of image plane
Nz = 51

x2 = np.linspace(-L_focal, L_focal, Nx)
# z2 = np.linspace(-6*L_focal, 6*L_focal , Nz)

E_x = np.zeros((Nx), dtype=np.complex64)
E_y = np.zeros((Nx), dtype=np.complex64)
E_z = np.zeros((Nx), dtype=np.complex64)

# normalization and steps of integral
N_theta = 100
N_phi = 100
delta_theta = alpha / N_theta
delta_phi = 2 * pi / N_phi

theta = 0
phi = 0

# Lineaire en x
P = np.array([1, 1, 0])

for n_theta in range(N_theta):
	cosTheta = np.cos(theta)
	sinTheta = np.sin(theta)
	tanTheta = np.tan(theta)
	cosThetaSqrt = np.sqrt(cosTheta)

	A = np.exp(-2 * f ** 2 * tanTheta ** 2 / w_centre ** 2)

	# juste la courrone
	#     A =  np.exp(-2*(f*tanTheta - d_courrone)**2/w_courrone**2)
	phi = 0

	for n_phi in range(N_phi):
		cosPhi = np.cos(phi)
		sinPhi = np.sin(phi)

		a = 1 + (cosTheta - 1) * (cosPhi) ** 2
		b = (cosTheta - 1) * cosPhi * sinPhi
		c = -sinTheta * cosPhi
		d = 1 + (cosTheta - 1) * (sinPhi) ** 2
		e = -sinTheta * sinPhi
		ff = cosTheta

		V = np.array([[a, b, c], [b, d, e], [-c, -e, ff]])

		# polarization in focal region
		PP = np.dot(V, P)

		# TODO j'ai enleve le 1j present dans le code de la thèse mais pas dans la formule.
		# terme_commun = 1j*sinTheta*cosThetaSqrt* np.exp(1j * k * (ZZ*cosTheta + sinTheta(XX*cosPhi + YY*sinPhi))) * delta_theta*delta_phi
		# terme_commun = A*sinTheta*cosThetaSqrt* np.exp(1j * k * (sinTheta*(x2*cosPhi))) * delta_theta*delta_phi
		terme_commun = 1j * A * sinTheta * cosThetaSqrt * np.exp(
			1j * k * (sinTheta * (x2 * cosPhi))) * delta_theta * delta_phi

		E_x += PP[0] * terme_commun
		E_y += PP[1] * terme_commun
		E_z += PP[2] * terme_commun

		phi += delta_phi

	theta += delta_theta

# print(E_x)
Ix = np.abs(E_x)
Iy = np.abs(E_y)
Iz = np.abs(E_z)

"""
#print(Ix)
plt.contourf(x2, y2, Ix[:,:,int(Nz/2)], 300)
plt.show()
plt.contourf(x2, y2, Ix[:,int(Ny/2),:], 300)
plt.show()
"""
# Iy2 = np.abs(Ey2)
# Iz2 = np.abs(Ez2)
I_z_equal_0 = Ix + Iy + Iz
I_z_equal_0 /= np.max(I_z_equal_0)
PSF_z_equal_zero = np.copy(I_z_equal_0)
# plt.plot(x2, I_z_equal_0)
# plt.show()


# profil en r
i = 0
z = 0
a = diam_confocal_pinhole / G
R = w
print(" a (µm) : ", a * 1E6)
print(" R (µm) : ", R * 1E6)
print(" w (µm) : ", w * 1E6)
# print("a : ", a)


confocal_filtering = np.zeros(Nx)

R = w
for x in x2:
	# R = w * np.sqrt(1 + (z * lambda_em_nm / (np.pi * w**2))**2)

	# print("R : ", R)
	# print("x : ", x)
	rho = abs(x)



	if rho >= R + a:
		confocal_filtering[i] = 0
		I_z_equal_0[i] = 0
		# print("x >= R + a : ", x, confocal_filtering[i])

	elif rho < R + a and rho > abs(R - a):
		if x != 0:
			arg_theta_1 = (a ** 2 + x ** 2 - R ** 2) / (2 * a * x)
			theta1 = np.arccos((a ** 2 + rho ** 2 - R ** 2) / (2 * a * rho))
			arg_theta_2 = (R ** 2 + x ** 2 - a ** 2) / (2 * R * x)
			theta2 = np.arccos((R ** 2 + rho ** 2 - a ** 2) / (2 * R * rho))
		else:
			theta1 = theta2 = 0  # ?

		arg_Delta = (a + rho + R) * (-a + rho + R) * (a - rho + R) * (a + rho - R)
		Delta = 0.5 * np.sqrt((a + rho + R) * (-a + rho + R) * (a - rho + R) * (a + rho - R))

		confocal_filtering[i] = (max(a, w)) ** 2 * (theta1 * a ** 2 + theta2 * R ** 2 - Delta) / (np.pi * a ** 2 * R ** 2)
		I_z_equal_0[i] *= confocal_filtering[i]
		# print("x < R + a and x > abs(R - a) : ", x, a, abs(R - a), confocal_filtering[i])

	elif rho <= abs(R - a):
		# Ici w = R donc le terme vaut toujours 1 (normalement)
		confocal_filtering[i] = (max(a, w)) ** 2 / (max(a, R)) ** 2
		I_z_equal_0[i] *= confocal_filtering[i]
		# print("x <= abs(R - a) : ", x, a, abs(R - a), confocal_filtering[i])

	i += 1


x_start = 0
x_stop = -1

i = 0
flag_peak = False
for x in x2:
	if not flag_peak  and I_z_equal_0[i] > threshold_MDF_detection:
		x_start = i
		flag_peak = True
	if flag_peak and I_z_equal_0[i] < threshold_MDF_detection:
		x_stop = i
	i += 1

x_dim = x2[x_stop] - x2[x_start]
print("x_dim (nm): ", x_dim*1E9)



plt.plot(x2 * 1E6, PSF_z_equal_zero, label="PSF NA=" + str(NA) + " pupill fill = " + str(w_centre))
plt.plot(x2 * 1E6, confocal_filtering, label="Confocal filter, pinhole = " + str(diam_confocal_pinhole_microns))
plt.plot(x2 * 1E6, I_z_equal_0, label="MDF")
plt.xlabel("x in µm")
plt.ylabel("MDF")
plt.title("Profil collection en x, dans le plan z=0")
plt.legend()
plt.show()






z2 = np.linspace(-exploration_factor_z * L_focal, exploration_factor_z * L_focal, Nz)

E_x = np.zeros((Nz), dtype=np.complex64)
E_y = np.zeros((Nz), dtype=np.complex64)
E_z = np.zeros((Nz), dtype=np.complex64)

theta = 0
phi = 0

for n_theta in range(N_theta):
	cosTheta = np.cos(theta)
	sinTheta = np.sin(theta)
	tanTheta = np.tan(theta)
	cosThetaSqrt = np.sqrt(cosTheta)

	A = np.exp(-2 * f ** 2 * tanTheta ** 2 / w_centre ** 2)
	phi = 0

	for n_phi in range(N_phi):
		cosPhi = np.cos(phi)
		sinPhi = np.sin(phi)

		a = 1 + (cosTheta - 1) * (cosPhi) ** 2
		b = (cosTheta - 1) * cosPhi * sinPhi
		c = -sinTheta * cosPhi
		d = 1 + (cosTheta - 1) * (sinPhi) ** 2
		e = -sinTheta * sinPhi
		ff = cosTheta

		V = np.array([[a, b, c], [b, d, e], [-c, -e, ff]])

		# polarization in focal region
		PP = np.dot(V, P)

		# TODO j'ai enleve le 1j present dans le code de la thèse mais pas dans la formule.
		# terme_commun = 1j*sinTheta*cosThetaSqrt* np.exp(1j * k * (ZZ*cosTheta + sinTheta(XX*cosPhi + YY*sinPhi))) * delta_theta*delta_phi
		# terme_commun = A*sinTheta*cosThetaSqrt* np.exp(1j * k * (sinTheta*(x2*cosPhi))) * delta_theta*delta_phi
		terme_commun = 1j * A * sinTheta * cosThetaSqrt * np.exp(1j * k * (z2 * cosTheta)) * delta_theta * delta_phi

		E_x += PP[0] * terme_commun
		E_y += PP[1] * terme_commun
		E_z += PP[2] * terme_commun

		phi += delta_phi

	theta += delta_theta

# print(E_x)
Ix = np.abs(E_x)
Iy = np.abs(E_y)
Iz = np.abs(E_z)

I_x_equal_0 = Ix + Iy + Iz
I_x_equal_0 /= np.max(I_x_equal_0)

PSF_x_equal_zero = np.copy(I_x_equal_0)

confocal_filtering = np.zeros(Nz)

# profil en z
i = 0
a = diam_confocal_pinhole / G

# print(" a : ", a)
# print(" R : ", R)
# print("a : ", a)
x = 0
for z in z2:
	R = w * np.sqrt(1 + (z * lambda_em / (np.pi * w ** 2)) ** 2)

	if x >= R + a:
		# print("x >= R + a : ", x, a)
		confocal_filtering[i] = 0
		I_x_equal_0[i] = 0

	elif x < R + a and x > abs(R - a):
		pass
	# print("x < R + a and x > abs(R - a) : ", x, a)
	# I_x_equal_0[i] *= (max(a, w))**2 * (theta1*a**2 + theta2*R**2 - Delta)/(np.pi * a**2 * R**2)
	elif x <= abs(R - a):
		confocal_filtering[i] = (max(a, w)) ** 2 / (max(a, R)) ** 2
		I_x_equal_0[i] *= confocal_filtering[i]

	i += 1

z_start = 0
z_stop = -1

i = 0
flag_peak = False
for z in z2:
	if not flag_peak and I_x_equal_0[i] > threshold_MDF_detection:
		z_start = i
		flag_peak = True
	if flag_peak and I_x_equal_0[i] < threshold_MDF_detection:
		z_stop = i
	i += 1

z_dim = z2[z_stop] - z2[z_start]
print("z_dim (µm) : ", z_dim*1E6)

plt.plot(z2 * 1E6, I_x_equal_0, label="MDF")
plt.plot(z2 * 1E6, PSF_x_equal_zero, label="PSF NA=" + str(NA) + " pupill fill = " + str(w_centre))
plt.plot(z2 * 1E6, confocal_filtering, label="Confocal filter, pinhole = " + str(diam_confocal_pinhole_microns))
plt.xlabel("z in µm")
plt.ylabel("MDF")
plt.title("Profil collection en z, dans le plan x=0")
plt.legend()
plt.show()



