# coding: utf-8

import numpy as np
import Sample

import matplotlib
from matplotlib import cm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from scipy.optimize import curve_fit

#TODO découper en bloc de temps pour limiter la RAM.
"""
#TODO script pour tester
- Angle
    - Intensité Moyenne
    - Tps de correlation
    - Taille Speckle
- Distance
    - Taille Speckle
    - Tps de correlation
- Taille des particules
- Taille du Volume
- Taille du detecteur
- Analyse Spectrale
"""

class DLS_Exp:
    """
    L'axe optique est selon Ox et le detecteur (si ponctuel) est placé dans le plan xy.
    """
    def __init__(self, sample, laserWl = 632.8, refractiveIndex=1.33, detecDistance_cm=5, detecAngle=45, detecSize_micron=50, deltaT_micro=1, totalTime_s=1):
        self.sp = sample
        self.deltaT_micro = deltaT_micro
        self.totalTime_s = totalTime_s
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

        #TODO light direction, polarization excitation et detection, Taille du pinhole.


    def getIntensityTimeTrace(self):

        nbTick = int(self.totalTime_s / (self.deltaT_micro*1E-6))
        #TODO meilleur nom pour I ?
        I = np.empty(nbTick)

        t = 0


        nbTickPerCalculation = 10000
        nbOfCalculationStep = int(nbTick/ nbTickPerCalculation)

        Xp = self.sp.particles[:]['x']
        Yp = self.sp.particles[:]['y']
        Zp = self.sp.particles[:]['z']

        posParticles = [Xp, Yp, Zp]
        posDetecteur = [self.x_d, self.y_d, 0]

        # for i in range(nbTick):
        #     # distPartDetec = np.sqrt((self.x_d - Xp) ** 2 + (self.y_d - Yp) ** 2 + (0 - Zp) ** 2)
        #     # chpElecParParticule = np.exp(1j * (self.k * Xp + self.k * distPartDetec))
        #     # chpElec = np.sum(chpElecParParticule)
        #     # I[i] = np.abs(chpElec)
        #     I[i] = np.abs( np.sum( np.exp(1j * (self.k*Xp - self.k*np.sqrt((self.x_d - Xp)** 2 + (self.y_d - Yp)** 2 + (0 - Zp)** 2)))))
        #     #I[i] = np.abs(np.sum(np.exp(1j * (self.k * Xp - self.k * np.linalg.norm(posParticles - posDetecteur)))))
        #
        #     self.sp.brownianMotionOneStep()
        #     t += self.deltaT_micro

        for i in range(nbOfCalculationStep):
            # distPartDetec = np.sqrt((self.x_d - Xp) ** 2 + (self.y_d - Yp) ** 2 + (0 - Zp) ** 2)
            # chpElecParParticule = np.exp(1j * (self.k * Xp + self.k * distPartDetec))
            # chpElec = np.sum(chpElecParParticule)
            # I[i] = np.abs(chpElec)

            # I[i] = np.abs(np.sum(np.exp(1j * (self.k * Xp - self.k * np.linalg.norm(posParticles - posDetecteur)))))

            mvtEvolution = self.sp.brownian_motion_multi_step(self.sp.particles, nbTickPerCalculation)

            Xp = mvtEvolution[:,:]['x']
            Yp = mvtEvolution[:,:]['y']
            Zp = mvtEvolution[:,:]['z']

            I[i*nbTickPerCalculation:(i+1)*nbTickPerCalculation] = np.abs(np.sum(np.exp(
                1j * (self.k * Xp - self.k * np.sqrt((self.x_d - Xp) ** 2 + (self.y_d - Yp) ** 2 + (0 - Zp) ** 2)))))

            self.sp.particles = mvtEvolution[-1]

            #t += self.deltaT_micro

        return I


        def autocorr(x, length):
            result = np.zeros(length)
            meanSquare = np.mean(x) ** 2  # * length/np.size(x)
            result[0] = np.sum(x ** 2)
            for j in range(1, length):
                result[j] = np.sum(x[0:-j] * x[j:])

            result /= np.size(x)
            result /= meanSquare
            return result

        autoCorrelationLength = 500

        # plt.plot(Intensity)
        time = np.linspace(0, autoCorrelationLength)
        time *= self.deltaT_micro * 1E-6
        photonCorrelation = autocorr(I, autoCorrelationLength)
        plt.plot(photonCorrelation)
        #plt.plot(I)
        plt.show()




        # # print (np.shape(listParticulePosition))
        # nbOfTimeClick = np.shape(self.brownianMotion.pos)[0]
        # nbreParticule = np.shape(self.brownianMotion.pos)[1]
        # timeTraceElecField = np.zeros(nbOfTimeClick)
        #
        # X = self.brownianMotion.pos[:, :, 0]
        # Y = self.brownianMotion.pos[:, :, 1]
        # Z = self.brownianMotion.pos[:, :, 2]
        #
        # # c'est un tableau qui fait ([tps], [numParticule] [DistanceAuDetecteur})
        # distanceOverTime = np.sqrt((self.x_d - X) ** 2 + (
        #     self.y_d - Y) ** 2 + (
        #                                self.z_d - Z) ** 2)
        # elecField = np.exp(1j * (self.k * X + self.k * distanceOverTime))
        # #elecField = np.exp(1j * (self.k * distanceOverTime))
        #
        # elecFieldTot = np.sum(elecField, axis=1)
        # Intensity = np.abs(elecFieldTot)
        #
        # def autocorr(x, length):
        #     result = np.zeros(length)
        #     meanSquare = np.mean(x) ** 2  # * length/np.size(x)
        #     result[0] = np.sum(x**2)
        #     for j in range(1, length):
        #         print(j)
        #         result[j] = np.sum(x[0:-j] * x[j:])
        #
        #     result /= np.size(x)
        #     result /= meanSquare
        #     return result
        #
        #
        # autoCorrelationLength = 500
        #
        # #plt.plot(Intensity)
        # time = np.linspace(0,autoCorrelationLength)
        # time *= self.deltaT_micro * 1E-6
        # photonCorrelation = autocorr(Intensity, autoCorrelationLength)
        # # plt.plot(photonCorrelation)
        # plt.plot(Intensity)
        # plt.show()
        #
        # #TODO Fit
        #
        # def expDecay(xdata, *params):
        #     tau = params[0]
        #     amplitude = params[1]
        #     return amplitude*np.exp(-xdata/tau)
        #
        # iniGuessForTau  = 1
        # iniGuessForAmplitude = 1
        # maxTau = 5000
        # maxAmplitude = 5000
        # # popt, pcov = curve_fit(f=expDecay, xdata=time, ydata=photonCorrelation, p0=[iniGuessForTau, iniGuessForAmplitude], bounds=(0, [maxTau, maxAmplitude]) )
        # # print('fit: tau=%5.3f, amplitude=%5.3f' % tuple(popt))
        #
        # # ps = np.abs(np.fft.fft(Intensity)) ** 2
        # #
        # # time_step = self.deltaT_micro * 1E-6
        # # freqs = np.fft.fftfreq(Intensity.size, time_step)
        # # idx = np.argsort(freqs)
        # #
        # # plt.plot(freqs[idx], ps[idx])
        # # plt.show()

    def getIntensityMap(self, mapSizeX=500, mapSizeY=500, pixelSizeX=1, pixelSizeY=1, timeIdx=0):
        """
        Unit is micrometer
        :param mapSizeX:
        :param mapSizeY:
        :param pixelSizeX:
        :param pixelSizeY:
        :return:
        """


        taillePixelMicron = 1
        NbrePixel = 500

        #Create Map memory
        mapElec = np.zeros((NbrePixel, NbrePixel), dtype=complex)
        map = np.zeros((NbrePixel, NbrePixel))
        y = np.arange(NbrePixel)
        z = np.arange(NbrePixel)

        z = self.z_d + z*taillePixelMicron
        y = self.y_d + y*taillePixelMicron

        YY, ZZ = np.meshgrid(y, z)  # Return coordinate matrices from coordinate vectors.
        # Le déphasage, approximée vaut r.q en produit scalaire.
        # Mais on peut garder la valeur exacte avec des ondes sphériques.

        # Quand on se balade sur le detecteur on change :
        #   - la distance par rapport aux particules
        #   - L'angle theta (mais tellement peu ?)

        # Liste des positions en y de toutes les particules au temps t=0 self.brownianMotion[0,:, 1]

        listParticulePosition = self.brownianMotion.pos[timeIdx, :, :]

        Xp = listParticulePosition[:, 0]
        Yp = listParticulePosition[:, 1]
        Zp = listParticulePosition[:, 2]

        nbreParticule = np.shape(listParticulePosition)[0]
        for i in range(nbreParticule):
            #rPartDetec = np.sqrt((XX - listParticulePosition[:, 0]) ** 2 + (YY - listParticulePosition[:, 1]) ** 2 + listParticulePosition[:, 2])
            rPartDetec = np.sqrt((self.x_d - Xp[i])**2 + (YY - Yp[i])**2  + (ZZ - Zp[i])**2)

            #POur l'instant je reste sur 3 possiblitiés poue le déphasage...

            chpElecParParticule = np.exp(1j * (self.k * listParticulePosition[i, 0] + self.k*rPartDetec))
            #chpElecParParticule = np.cos(self.k*rPartDetec + self.k * listParticulePosition[i, 0])

            #chpElecParParticule = np.cos(self.k * rPartDetec)
            #chpElecParParticule = np.exp(1j * (self.k * rPartDetec))


            #TODO marche pas encore
            #chpElecParParticule = np.exp(1j * (self.q_x * Xp[i] + self.q_y * Yp[i]))

            # cp = plt.contourf(y, z, np.abs(chpElecParParticule), 30, cmap=cm.hot)
            # plt.colorbar(cp)
            # plt.show()

            mapElec += chpElecParParticule

        #imageChpElec =  np.cumsum(chpElecParParticule)

        map = np.abs(mapElec)

        cp = plt.contourf(y, z, map, 30, cmap=cm.hot)
        plt.colorbar(cp)
        plt.show()

    def getIntensityMapMovie(self, mapSizeX=500, mapSizeY=500, pixelSizeX=1, pixelSizeY=1, numTimeStep=10):

        fig = plt.figure(figsize=(8, 6))
        nbFrame = 20

        posDecteurCm = np.array([5, 5, 0])
        posDetecteur = posDecteurCm * 1000

        taillePixelMicron = 1
        NbrePixel = 500

        # Create Map memory
        map = np.zeros((NbrePixel, NbrePixel))
        mapElec = np.zeros((NbrePixel, NbrePixel), dtype=complex)
        y = np.arange(NbrePixel)
        z = np.arange(NbrePixel)
        z = posDetecteur[2] + z * taillePixelMicron
        y = posDetecteur[1] + y * taillePixelMicron

        YY, ZZ = np.meshgrid(y, z)  # Return coordinate matrices from coordinate vectors.


        listParticulePosition = self.brownianMotion.pos[0, :, :]
        #print(listParticulePosition)
        #print (np.shape(listParticulePosition))
        nbreParticule = np.shape(listParticulePosition)[0]


        # Pour installer conda install -c conda-forge ffmpeg
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=5, metadata=metadata)

        idxT = 0

        with writer.saving(fig, "Speckle.mp4", nbFrame):
            for i in range(nbFrame):
                print("frame :", i)
                map.fill(0)
                mapElec.fill(0)
                listParticulePosition = self.brownianMotion.pos[idxT, :, :]
                for i in range(nbreParticule):
                    rPartDetec = np.sqrt((posDetecteur[0] - listParticulePosition[i, 0]) ** 2 + (
                        YY - listParticulePosition[i, 1]) ** 2 + (
                                             ZZ - listParticulePosition[i, 2]) ** 2)


                    #chpElecParParticule = np.exp(1j * (self.k * listParticulePosition[i, 0] + self.k * rPartDetec))
                    chpElecParParticule = np.exp(1j * (self.k * listParticulePosition[i, 0] + self.k * rPartDetec))
                    # chpElecParParticule = np.cos(self.k*rPartDetec + self.k * listParticulePosition[i, 0])

                    mapElec += chpElecParParticule

                    # L'onde va mettre le temps t_transfert = x/c pour atteindre la particule dont l'abscisse est x
                    # Y avait-il une onde ?
                    # Il faut pour cela que ce teps r/c soit compris entre t_transfert et t_transfert + dureeImpulsion.

                map = np.abs(mapElec)
                ax = plt.gca()

                cp = plt.contourf(y, z, map, 30, cmap=cm.hot)
                writer.grab_frame()
                idxT += 1


if __name__ == "__main__":

    def testDeBase():
        sp = Sample.Sample(nbParticle=100, dimXYZ_micron=(100, 100, 100), particleIniConfig="random")
        dls = DLS_Exp(sp, detecDistance_cm=3, deltaT_micro=0.1, totalTime_s=0.1)
        # dls.getIntensityMap()
        # dls.getIntensityMapMovie()
        dls.getIntensityTimeTrace()
        print("OK")


    def testDependanceAngle(self, thetaMin = 0, thetaMax=90, nbStep=5):

        def autocorr(x, length):
            result = np.zeros(length)
            meanSquare = np.mean(x) ** 2  # * length/np.size(x)
            result[0] = np.sum(x ** 2)
            for j in range(1, length):
                result[j] = np.sum(x[0:-j] * x[j:])

            result /= np.size(x)
            result /= meanSquare
            return result



        autoCorrelationLength = 500
        time = np.linspace(0, autoCorrelationLength)
        time *= deltaT_micro_ * 1E-6


        thetas = np.linspace(thetaMin * np.pi / 180, thetaMax * np.pi / 180, nbStep)

        deltaT_micro_ = 0.1

        listIntensity = []

        for theta in thetas:
            sp = Sample.Sample(nbParticle=100, dimXYZ_micron=(100, 100, 100), particleIniConfig="random")
            dls = DLS_Exp(sp, detecDistance_cm=5, deltaT_micro=deltaT_micro_, totalTime_s=0.1, detecAngle=theta)
            I = dls.getIntensityTimeTrace()
            photonCorrelation = autocorr(I, autoCorrelationLength)
            plt.plot(photonCorrelation, time, label="%f" % theta)

        plt.show()




    testDependanceAngle()


