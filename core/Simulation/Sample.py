# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class Sample:
    def __init__(self, nbParticle=10, dimXYZ_micron=(100,100,100), deltaT_micro=1, temperature=300, viscosity=1E-3, particleIniConfig="random"):
        """
        Cette classe gere le mouvements des particules au sein de la  boite de simulation.
        On ne prend pas en compte ici l'interraction lumière matière.
        Ne pouvant pas iterer facilement avec des boucles for, on travaille avec des gros blocs temporelles pouvant
        occuper 1Go dans la RAM.
        :param nbParticle:
        :param dimXYZ_micron:
        :param deltaT_micro:
        :param temperature:
        :param viscosity:
        :param particleIniConfig:
        """

        self.nb_particle = nbParticle
        self.particles = None
        self.diff_coeff = None
        self.maxRAM_inMo = 500
        self.deltaT_micro = deltaT_micro
        self.temperature = temperature
        self.viscosity = viscosity
        self.dimX, self.dimY, self.dimZ = dimXYZ_micron

        # self.npParticleType = np.dtype([('x', np.float), ('y', np.float), ('z', np.float), ('phi1', np.float), ('phi2', np.float), ('phi3', np.float), ('Dtx', np.float), ('Dty', np.float), ('Dtz', np.float), ('Drx', np.float), ('Dry', np.float), ('Drz', np.float),  ('axx', np.float), ('axy', np.float), ('axz', np.float), ('ayx', np.float), ('ayy', np.float), ('ayz', np.float), ('azx', np.float), ('azy', np.float), ('azz', np.float)])
        self.npParticleType = np.dtype([('x', np.float), ('y', np.float), ('z', np.float), ('Dtx', np.float), ('Dty', np.float), ('Dtz', np.float)])
        self.create_particles()
        self.set_particles_ini_position(particleIniConfig)




    def create_particles(self):
        self.particles = np.zeros(self.nb_particle, dtype=self.npParticleType)
        #TODO lire depuis un fichier les caracteristiques des particules
        # TODO how to create a list of particle : form file ? Algo ?

        # Taille, distribution sur une gaussienne,

        # Pour l'instant particules spherique
        self.particles[:]['Dtx'] = self.particles[:]['Dtz'] = self.particles[:]['Dty'] = 1E-9
        # self.particles[:]['Drx'] = self.particles[:]['Drz'] = self.particles[:]['Dry'] = 1E-9
        # self.particles[:]['axx'] = self.particles[:]['ayy'] = self.particles[:]['azz'] = 1

    def set_particles_ini_position(self, mode='random'):
        if mode == "random":
            r = np.random.rand(self.nb_particle, 3)
            self.particles[:]['x'] = r[:, 0] * self.dimX
            self.particles[:]['y'] = r[:, 1] * self.dimY
            self.particles[:]['z'] = r[:, 2] * self.dimZ

            #self.particles[:]['phi1'], self.particles[:]['phi2'], self.particles[:]['phi3'] = np.random.rand(self.nbParticle, 3) * 2 * np.pi
        elif mode == "crystal":
            self.d_Bragg = self.dimX * 1.0 / self.nb_particle
            self.particles[:]['x'] = np.linspace(0, self.dimX, self.nb_particle)
            self.particles[:]['y'] = np.linspace(0, self.dimY, self.nb_particle)
            self.particles[:]['z'] = np.linspace(0, self.dimZ, self.nb_particle)
            pass

    # def brownian_motion_one_step(self):
    #     #TODO verifier que c'est équivalent de tirer sur une gaussienne de largeur 1 puis de multiplier par δL, ou de tirer sur une gaussienne de largeur δL
    #
    #     dr = np.random.normal(loc=0, scale=1.0, size=(self.nb_particle, 3))
    #
    #     #TODO opt : 2 * Delta T pré-calculé.
    #     self.particles[:]['x'] += dr[:, 0] * np.sqrt(2 * self.particles[:]['Dtx'] * self.deltaT_micro * 1E-6) * 1E6
    #     self.particles[:]['y'] += dr[:, 1] * np.sqrt(2 * self.particles[:]['Dty'] * self.deltaT_micro * 1E-6) * 1E6
    #     self.particles[:]['z'] += dr[:, 2] * np.sqrt(2 * self.particles[:]['Dtz'] * self.deltaT_micro * 1E-6) * 1E6

    def brownian_motion(self, pos_ini, nbOfStep=1000):
        # Draw random samples from a normal (Gaussian) distribution.
        dr = np.random.normal(loc=0, scale=1.0, size=(nbOfStep, self.nb_particle, 3))

        # Construct the brownian trajectory by adding all the displacement
        dr = np.cumsum(dr, axis=0, out=dr)

        # TODO do not create a new array at each iteration
        mvt_evolution = np.zeros((nbOfStep, self.nb_particle), dtype=self.npParticleType)

        # offsetting at t=0 by the initial position
        mvt_evolution[:] = pos_ini

        # Scaling the displacement with the diffusion coefficient
        mvt_evolution[:]['x'] += dr[:, :, 0] * np.sqrt(2 * self.particles[:]['Dtx'] * self.deltaT_micro * 1E-6) * 1E6
        mvt_evolution[:]['y'] += dr[:, :, 1] * np.sqrt(2 * self.particles[:]['Dty'] * self.deltaT_micro * 1E-6) * 1E6
        mvt_evolution[:]['z'] += dr[:, :, 2] * np.sqrt(2 * self.particles[:]['Dtz'] * self.deltaT_micro * 1E-6) * 1E6

        # TODO conditions périodiques aux limites.

        return mvt_evolution


    def plot_brownian_2D_trajectory(self, nb_time_step=200):
        x = np.zeros(nb_time_step)
        y = np.zeros(nb_time_step)

        for i in range(nb_time_step):
            x[i] = self.particles[0]['x']
            y[i] = self.particles[0]['y']
            self.brownian_motion_one_step()


        # Plot the 2D trajectory.
        plt.plot(x, y)

        # Mark the start and end points.
        plt.plot(x[0], y[0], 'go')
        plt.plot(x[-1], y[-1], 'ro')
        plt.show()

    def histo_test(self, nbTimeStep_=200):
        nbTimeStep = nbTimeStep_
        x = np.zeros(nbTimeStep)

        for i in range(nbTimeStep):
            x[i] = self.particles[0]['x']
            self.brownian_motion_one_step()


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
    sp.brownian_motion(pos_ini=sp.particles, nbOfStep=3)
    sp.plot_brownian_2D_trajectory()

