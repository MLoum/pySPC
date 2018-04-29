# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class Sample:
    def __init__(self, nbParticle=10, dimXYZ_micron=(100,100,100), deltaT_micro=1, temperature = 300, viscosity = 1E-3, particleIniConfig="random"):
        # self.pos contient les positions au cours du temps sous la forme
        # size=(nbOfTimeStep, self.nbParticle, 3)
        # i.e temps, num Position, x y ou z

        self.pos = None

        self.nbParticle = nbParticle
        self.particles = None
        self.diffCoeff = None
        self.maxRAM_inMo = 500
        self.deltaT_micro = deltaT_micro
        self.temperature = temperature
        self.viscosity = viscosity
        self.dimX, self.dimY, self.dimZ = dimXYZ_micron

        self.npParticleType = np.dtype([('x', np.float), ('y', np.float), ('z', np.float), ('phi1', np.float), ('phi2', np.float), ('phi3', np.float), ('Dtx', np.float), ('Dty', np.float), ('Dtz', np.float), ('Drx', np.float), ('Dry', np.float), ('Drz', np.float),  ('axx', np.float), ('axy', np.float), ('axz', np.float), ('ayx', np.float), ('ayy', np.float), ('ayz', np.float), ('azx', np.float), ('azy', np.float), ('azz', np.float)])

        self.createParticles()
        self.setParticlesIniPosition(particleIniConfig)


    def createParticles(self):
        self.particles = np.zeros(self.nbParticle, dtype=self.npParticleType)
        #TODO lire depuis un fichier les caracteristiques des particules
        # Taille, distribution sur une gaussienne,

        #Pour l'instant une sphere

        self.particles[:]['Dtx'] = self.particles[:]['Dtz'] = self.particles[:]['Dty'] = 1E-9
        self.particles[:]['Drx'] = self.particles[:]['Drz'] = self.particles[:]['Dry'] = 1E-9
        self.particles[:]['axx'] = self.particles[:]['ayy'] = self.particles[:]['azz'] = 1

    def setParticlesIniPosition(self, mode='random'):
        if mode == "random":
            r =  np.random.rand(self.nbParticle, 3)
            self.particles[:]['x'] = r[:, 0] * self.dimX
            self.particles[:]['y'] = r[:, 1] * self.dimY
            self.particles[:]['z'] = r[:, 2] * self.dimZ

            #self.particles[:]['phi1'], self.particles[:]['phi2'], self.particles[:]['phi3'] = np.random.rand(self.nbParticle, 3) * 2 * np.pi
        elif mode == "crystal":
            self.d_Bragg = self.dimX * 1.0 / self.nbParticle
            self.particles[:]['x'] = np.linspace(0, self.dimX, self.nbParticle)
            self.particles[:]['y'] = np.linspace(0, self.dimY, self.nbParticle)
            self.particles[:]['z'] = np.linspace(0, self.dimZ, self.nbParticle)
            pass

    def brownianMotionOneStep(self):
        #TODO verifier que c'est équivalent de tirer sur une gaussienne de largeur 1 puis de multiplier par δL, ou de tirer sur une gaussienne de largeur δL

        dr = np.random.normal(loc=0, scale=1.0, size=(self.nbParticle, 3))

        #TODO opt : 2 * Delta T pré-calculé.
        self.particles[:]['x'] += dr[:, 0] * np.sqrt(2 * self.particles[:]['Dtx'] * self.deltaT_micro * 1E-6) * 1E6
        self.particles[:]['y'] += dr[:, 1] * np.sqrt(2 * self.particles[:]['Dty'] * self.deltaT_micro * 1E-6) * 1E6
        self.particles[:]['z'] += dr[:, 2] * np.sqrt(2 * self.particles[:]['Dtz'] * self.deltaT_micro * 1E-6) * 1E6

    def brownianMotionMultiStep(self, evolIni, nbOfStep=1000):
        dr = np.random.normal(loc=0, scale=1.0, size=(nbOfStep, self.nbParticle, 3 ))

        dr = np.cumsum(dr, axis=0, out=dr)

        mvtEvolution = np.zeros((nbOfStep, self.nbParticle), dtype=self.npParticleType)
        mvtEvolution[:] = evolIni
        mvtEvolution[:]['x'] += dr[:, :, 0] * np.sqrt(2 * self.particles[:]['Dtx'] * self.deltaT_micro * 1E-6) * 1E6
        mvtEvolution[:]['y'] += dr[:, :, 1] * np.sqrt(2 * self.particles[:]['Dty'] * self.deltaT_micro * 1E-6) * 1E6
        mvtEvolution[:]['z'] += dr[:, :, 2] * np.sqrt(2 * self.particles[:]['Dtz'] * self.deltaT_micro * 1E-6) * 1E6



        return mvtEvolution



    def setParticleCharacteristics(self):
        #3 pour Dx, Dy, Dz
        self.deltaL = np.ones( (self.nbParticle, 3))
        D = 1E-9
        self.deltaL *= np.sqrt(6 * D * self.deltaT_micro* 1E-6)

        #Conversion en micron
        self.deltaL *= 1E6


        self.posIni = np.random.rand(self.nbParticle, 3)

        self.posIni[:,:] *= [self.dimX, self.dimY, self.dimZ]


    def moveParticle(self, nbOfTimeStep=50):
        """
        De façon équivalente on peut tirer au hasard sur une sur une gaussienne centrée en 0 et de largeur δL =
        √6Dδt le rayon de la sphère où la particule se retrouve après la diffusion.
        La direction de diffusion est choisie aléatoirement. On peut pour cela tirer au hasard sur une gaussienne
        centrée en 0 et de largeur 1 trois coordonées d’un vecteur u : uδx, puis uδx, puis uδz. On a alors :
        x = x+δL uδx / ||u||
        y = x+δL uδy / ||u||
        y = x+δL uδz / ||u||

        :return:
        """
        #3 for translation x,y,z

        #TODO convection et sedimentation (un peu la même chose...)

        #Draw random samples from a normal (Gaussian) distribution.
        self.pos = np.random.normal(loc=0, scale=1.0, size=(nbOfTimeStep, self.nbParticle, 3))
        # NB : x,y,z  at time step 10 of particle 2 self.pos[10, 2, :]

        normRandomVector =  np.sqrt(self.pos[:, :, 0]**2 + self.pos[:, :, 1]**2 + self.pos[:, :, 2]**2)

        # self.pos[:,:,:] to prevent a copy of the data
        #TODO one liner ?
        self.pos[:, :, 0] *= self.deltaL[:, 0] / normRandomVector
        self.pos[:, :, 1] *= self.deltaL[:, 1] / normRandomVector
        self.pos[:, :, 2] *= self.deltaL[:, 2] / normRandomVector

        #At this step, self.pos contain all the displacements but not yet the position of the particle.

        #offsetting at t=0 by the (random) initial position
        self.pos[0, :, :] += self.posIni[:,:]

        #print(np.shape(self.pos))

        #construct the brownian trajectory by adding all the displacement
        # Je ne vois pas pourquoi la somme doit se faire selon l'axe 0, moi j'aurais dis 2, mais bon cela semble marcher
        np.cumsum(self.pos, axis=0, out = self.pos)

        #Conversion en micron ?
        #self.pos *= 1E6

        #TODO conditions périodiques aux limites.


    def testBrownianMotion(self):
        # Plot the 2D trajectory.
        plt.plot(self.pos[:,0,0], self.pos[:,0,1])

        # Mark the start and end points.
        plt.plot(self.pos[0,0,0],  self.pos[0,0,1], 'go')
        plt.plot(self.pos[-1,0,0], self.pos[-1,0,1], 'ro')
        plt.show()

    def testBrownianMotionNew(self, nbTimeStep_=200):

        nbTimeStep = nbTimeStep_
        x = np.zeros(nbTimeStep)
        y = np.zeros(nbTimeStep)

        for i in range(nbTimeStep):
            x[i] = self.particles[0]['x']
            y[i] = self.particles[0]['y']
            self.brownianMotionOneStep()


        # Plot the 2D trajectory.
        plt.plot(x, y)

        # Mark the start and end points.
        plt.plot(x[0], y[0], 'go')
        plt.plot(x[-1], y[-1], 'ro')
        plt.show()

    def histoTest(self, nbTimeStep_=200):
        nbTimeStep = nbTimeStep_
        x = np.zeros(nbTimeStep)

        for i in range(nbTimeStep):
            x[i] = self.particles[0]['x']
            self.brownianMotionOneStep()


        # et non np.diff(x) **2
        deltaX_square = np.diff(x**2)
        plt.hist(deltaX_square, bins = 50)
        plt.show()

if __name__ == "__main__":
    # sp = Sample(nbParticle=2)
    # sp.setParticleCharacteristics()
    # sp.moveParticle()
    # sp.testBrownianMotion()
    sp = Sample(nbParticle=2)
    sp.brownianMotionMultiStep(sp.particles, 3)
    #sp.testBrownianMotionNew(10000)
    sp.histoTest(10000)

