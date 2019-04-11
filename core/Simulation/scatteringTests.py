# coding: utf-8

import numpy as np
import Sample

import matplotlib
from matplotlib import cm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from scipy.optimize import curve_fit


class scatteringTests:
    """
    L'axe optique est selon Ox et le detecteur (si ponctuel) est placé dans le plan xy.
    """
    def __init__(self, sample,  laserWl = 632.8, refractiveIndex=1.33, detecDistance_cm=5, detecAngle=45, detecSize_micron=50):
        self.sp = sample
        self.laserWl = laserWl/1000.0   #µm
        self.refractiveIndex = refractiveIndex
        self.k = 2 * np.pi * refractiveIndex/ self.laserWl
        self.detecDistance_cm = detecDistance_cm
        self.detecSize_micron = detecSize_micron
        self.detecAngle = detecAngle * np.pi/180.0
        self.x_d = detecDistance_cm*10000*np.cos(self.detecAngle)
        self.y_d = detecDistance_cm*10000*np.sin(self.detecAngle)
        self.z_d= 0

        self.q_x = self.k * (1 - np.cos(self.detecAngle))
        self.q_y = -self.k * np.sin(self.detecAngle)

    def getIntensityMap(self, mapSizeX=500, mapSizeY=500, pixelSizeX=1, pixelSizeY=1, timeIdx=0):

        taillePixelMicron = pixelSizeX
        NbrePixel = mapSizeX

        # Create Map memory
        mapElec = np.zeros((NbrePixel, NbrePixel), dtype=complex)
        map = np.zeros((NbrePixel, NbrePixel))
        y = np.arange(NbrePixel)
        z = np.arange(NbrePixel)

        z = self.z_d + z * taillePixelMicron
        y = self.y_d + y * taillePixelMicron

        YY, ZZ = np.meshgrid(y, z)  # Return coordinate matrices from coordinate vectors.
        # Le déphasage, approximée vaut r.q en produit scalaire.
        # Mais on peut garder la valeur exacte avec des ondes sphériques.

        # Quand on se balade sur le detecteur on change :
        #   - la distance par rapport aux particules
        #   - L'angle theta (mais tellement peu ?)

        Xp = self.sp.particles[:]['x']
        Yp = self.sp.particles[:]['y']
        Zp = self.sp.particles[:]['z']

        #Au vu des dimensions des tableaux j'ai deux choix,
        # faire nbParticule Image et les sommer
        # faire une boucle sur les pixels


        for i in range(NbrePixel):
            for j in range(NbrePixel):
                #TODO Use numpy.linalg.norm: dist = numpy.linalg.norm(a-b)
                distPartDetec = np.sqrt((self.x_d - Xp) ** 2 + (y[i] - Yp) ** 2 + (z[j] - Zp) ** 2)
                chpElecParParticule = np.exp(1j * (self.k * Xp + self.k * distPartDetec))
                mapElec[i,j] = np.sum(chpElecParParticule)

        # nbreParticule = np.shape(Xp)[0]
        # for i in range(nbreParticule):
        #     rPartDetec = np.sqrt((self.x_d - Xp[i]) ** 2 + (YY - Yp[i]) ** 2 + (ZZ - Zp[i]) ** 2)
        #     chpElecParParticule = np.exp(1j * (self.k * Xp[i] + self.k * rPartDetec))
        #     mapElec += chpElecParParticule



        map = np.abs(mapElec)

        cp = plt.contourf(y, z, map, 30, cmap=cm.hot)
        plt.colorbar(cp)
        plt.show()


    def testThetaDependency(self, thetaMin = -180, thetaMax=180, nbStep=360, detecDistance_cm = 10, angIncidence = 0):
        thetas = np.linspace(thetaMin*np.pi/180, thetaMax*np.pi/180, nbStep)
        x_d = detecDistance_cm*10000*np.cos(thetas) + self.sp.dimX
        y_d = detecDistance_cm*10000*np.sin(thetas) + self.sp.dimY
        z_d = 0
        kx = self.k * np.cos(angIncidence*np.pi/180.0)
        ky = self.k * np.sin(angIncidence*np.pi/180.0)


        Xp = self.sp.particles[:]['x']
        Yp = self.sp.particles[:]['y']
        Zp = self.sp.particles[:]['z']

        I_theta = np.zeros(nbStep)


        for i in range(nbStep):
            print(i)
            distPartDetec = np.sqrt((x_d[i] - Xp) ** 2 + (y_d[i] - Yp) ** 2 + (z_d - Zp) ** 2)
            I_theta[i] = np.abs(np.sum(np.exp(1j * (kx*Xp + ky*Yp + self.k*distPartDetec))))


        #Avec un crystal, les longueurs d'ondes de la lumière sont bien trop grande pour avoir des angles de bragg.

        #angBragg = np.arcsin(self.laserWl / (2 * self.sp.d_Bragg)) * 180/np.pi
        print ("angle Incidence %f", angIncidence)
        #print ("angle Bragg %f", angBragg)

        plt.plot(thetas*180/np.pi, I_theta)
        plt.show()



if __name__ == "__main__":
    #sp = Sample.Sample(nbParticle=10000, dimXYZ_micron=(1000,1000,1000), particleIniConfig="crystal")
    sp = Sample.Sample(nbParticle=30000, dimXYZ_micron=(1000, 1000, 1000), particleIniConfig="random")
    #sp.testBrownianMotionNew(10000)
    scatTest = scatteringTests(sp)
    #scatTest.getIntensityMap()
    scatTest.testThetaDependency(angIncidence=0)