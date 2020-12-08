import numpy as np
from scipy.stats import rv_discrete
import multiprocessing as mp
from core.analyze.lifetime import IRF
import shelve
from scipy.ndimage.interpolation import shift as shift_scipy

class Decay:
    """
    A very simple and basic object to store the data of a decay and its parameter (lifetimes, amplitudes, ...)
    """

    def __init__(self, data, fitted_params_dict, aux_params_dict):
        self.data = data
        # paramater obtained from fit (tau, amplitude)
        self.fitted_params_dict = fitted_params_dict
        # parameter used to generate the data (noise, etc)
        self.aux_params_dict = aux_params_dict

class DecayGenerator():
    def __init__(self, nb_decay=100, time_nbins=4096, time_step_ns=0.01, model="single_exp", params_dict={}):
        self.nb_decay = nb_decay
        #FIXME
        self.time_nbins = time_nbins
        self.time_step_ns = time_step_ns
        self.model = model
        self.params_dict = params_dict

        self.time_idx = np.arange(self.time_nbins)  # time axis in index units
        self.time_ns = self.time_idx * self.time_step_ns  # time axis in nano-seconds

        self.decays = []
        self.irf = None

    def generate_data(self):
        decay_data = []
        if self.model == "single_exp":
            min_tau = self.params_dict["tau"][0]
            max_tau = self.params_dict["tau"][1]
            t0 = self.params_dict["t0"][0]
            noise_min = self.params_dict["noise"][0]
            noise_max = self.params_dict["noise"][1]
            nb_photon_min = self.params_dict["nb_photon"][0]
            nb_photon_max = self.params_dict["nb_photon"][1]
            tau_irf = self.params_dict["tau_irf"][0]
            shift_irf = self.params_dict["shift_irf"][0]

            # Tirer au hasard les parametres
            taus = np.random.uniform(min_tau, max_tau, size=self.nb_decay)
            noises = np.random.uniform(noise_min, noise_max, size=self.nb_decay)
            nb_photons = np.random.uniform(nb_photon_min, nb_photon_max, size=self.nb_decay).astype(np.int)

            is_multi_core = False
            if is_multi_core:
                # def generate_single_exp_decay_fct(tau, t0, noise, nb_photons):
                #     return self.generate_single_exp_decay(tau, t0, noise, nb_photons)

                #FIXME
                nb_of_workers = 4
                p = mp.Pool(nb_of_workers)
                self.decays = [
                    p.apply(self.generate_single_exp_decay, args=(taus[i], t0, noises[i], nb_photons[i], tau_irf))
                    for i in range(self.nb_decay)]
            else:
                # Single core
                self.decays = []
                for i in range(self.nb_decay):
                    # TODO vectorization ? Mais avec des parametres tous differents ?
                    self.decays.append(self.generate_single_exp_decay(taus[i], t0, noises[i], nb_photons[i], tau_irf))


    def generate_single_exp_decay(self, tau, t0, noise, nb_of_generated_photon, tau_irf=None, irf_shift=None):
        decay = np.exp(-(self.time_ns - t0) / tau)
        decay[self.time_ns < t0] = 0
        decay /= decay.sum()

        if tau_irf is not None:
            self.irf = IRF()
            params_dict = {"tau": tau_irf, "t0": t0, "irf_shift": irf_shift, "time_step_ns": self.time_step_ns, "nb_of_microtime_channel": self.time_nbins}
            self.irf.generate(params_dict, algo="Becker")
            # irf = (self.time_ns - t0) / tau_irf * np.exp(-(self.time_ns - t0) / tau_irf)
            # irf[irf < 0] = 0
            # if irf_shift is not None:
            #     irf = shift_scipy(irf, irf_shift, mode='wrap')
            # irf /= irf.sum()


            decay = np.convolve(decay, self.irf.processed_data)[0:np.size(decay)]
            decay /= decay.sum()

        decay_obj = rv_discrete(name='mono_exp', values=(self.time_idx, decay))
        photons = decay_obj.rvs(size=nb_of_generated_photon)
        decay_data = np.bincount(photons)
        nb_of_pad_data = self.time_idx.size - decay_data.size
        zeros = np.zeros(nb_of_pad_data)
        decay_data = np.concatenate((decay_data, zeros))

        decay_data += np.random.random(self.time_idx.size) * noise
        decay_data = decay_data.astype(np.int)
        aux_params_dict = {}
        fitted_params_dict = {}
        fitted_params_dict["tau"] = tau
        aux_params_dict["noise"] = noise
        aux_params_dict["nb_photon"] = nb_of_generated_photon

        return Decay(decay_data, fitted_params_dict, aux_params_dict)

    def generate_double_exp_decay(self, a1, a2, tau1, tau2, t0, noise, nb_of_generated_photon):
        C = 1 / (a1 * tau1 + a2 * tau2)
        decay = C * (a1 * np.exp(-(self.time_ns - t0) / tau1) + a2 * np.exp(-(self.time_ns - t0) / tau2))
        decay[self.time_ns < t0] = 0
        decay /= decay.sum()

        decay_obj = rv_discrete(name='biexpconv', values=(self.time_idx, decay))

        decay_data = decay_obj.rvs(size=nb_of_generated_photon) + np.random.random(self.time_idx.size) * noise
        return decay_data

    def save(self):
        #TODO with shelves
        pass

    def load(self):
        pass

    def plot_test(self):
        pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    params_dict_ = {}
    # Min / max
    params_dict_["tau"] = [1, 10]
    params_dict_["noise"] = [0, 0]
    params_dict_["nb_photon"] = [1000, 100000]
    params_dict_["t0"] = [0.5, 0.5]
    params_dict_["tau_irf"] = [0.5, 0.5]
    params_dict_["shift_irf"] = [0, 0]
    decay_generator = DecayGenerator(nb_decay=5, time_nbins=4096, model="single_exp", params_dict=params_dict_)
    decay_generator.generate_data()


    for decay in decay_generator.decays:
        plt.semilogy(decay_generator.time_ns, decay.data, label="tau = " + str(decay.fitted_params_dict["tau"])[0:4] + " ns")

    plt.legend()
    plt.xlabel("temps / ns")
    plt.ylabel("Nbre de photon")
    plt.title("Exemple de déclins mono-exponentiel")
    plt.savefig("declins.png", dpi=300)
    plt.show()
