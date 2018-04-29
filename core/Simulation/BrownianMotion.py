# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class BrownianMotion:
    def __init__(self, nbParticle=10, dimXYZ_micron=(100,100,100), deltaT_micro=1, temperature = 300, viscosity = 1E-3):
        # self.pos contient les positions au cours du temps sous la forme
        # size=(nbOfTimeStep, self.nbParticle, 3)
        # i.e temps, num Position, x y ou z

        self.pos = None

        self.nbParticle = nbParticle
        self.diffCoeff = None
        self.maxRAM_inMo = 500
        self.deltaT_micro = deltaT_micro
        self.temperature = temperature
        self.viscosity = viscosity
        self.dimX, self.dimY, self.dimZ = dimXYZ_micron

    def defineParticleCharacteristics(self):
        #3 pour Dx, Dy, Dz
        self.deltaL = np.ones( (self.nbParticle, 3))
        D = 1E-9
        self.deltaL *= np.sqrt(6 * D * self.deltaT_micro* 1E-6)

        #Conversion en micron
        self.deltaL *= 1E6


        self.posIni = np.random.rand(self.nbParticle, 3)

        self.posIni[:,:] *= [self.dimX, self.dimY, self.dimZ]


    def move(self, nbOfTimeStep=50):
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


    def plotTest(self):
        # Plot the 2D trajectory.
        plt.plot(self.pos[:,0,0], self.pos[:,0,1])

        # Mark the start and end points.
        plt.plot(self.pos[0,0,0],  self.pos[0,0,1], 'go')
        plt.plot(self.pos[-1,0,0], self.pos[-1,0,1], 'ro')
        plt.show()

    def histoTest(self):
        pass

if __name__ == "__main__":
    bm = BrownianMotion(nbParticle=2)
    bm.defineParticleCharacteristics()
    bm.move()
    bm.plotTest()