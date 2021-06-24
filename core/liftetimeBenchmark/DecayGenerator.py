import numpy as np
from scipy.stats import rv_discrete
import multiprocessing as mp
from core.analyze.lifetime import IRF
import shelve
from scipy.ndimage.interpolation import shift as shift_scipy
from core.analyze.lifetime import lifeTimeMeasurements


class DecayGenerator():
    def __init__(self, params_dict):
        self.nb_decay = params_dict["nb_decays"]
        #FIXME
        self.time_nbins = params_dict["nb_time_bins"]
        self.time_step_ns = params_dict["time_bins_duration"]/1000.0    #user interface is in ps
        self.model = params_dict["model_name"]
        self.params_dict = params_dict

        self.time_idx = np.arange(self.time_nbins)  # time axis in index units
        self.time_ns = self.time_idx * self.time_step_ns  # time axis in nano-seconds

        self.decays = []

        #TODO as many meseauremnt as workers.
        self.lifetime_measurement = lifeTimeMeasurements()
        self.irf = None

    def generate_data(self, params_dict):
        self.lifetime_measurement.set_model(params_dict["model_name"])

        dict_random_parameters = {}
        for key in self.lifetime_measurement.params.keys():
            if self.lifetime_measurement.params[key].user_data is not None:
                if "dontGenerate" in self.lifetime_measurement.params[key].user_data:
                    continue
            dict_random_parameters[key] = np.random.uniform(params_dict[key]["min_gen"], params_dict[key]["max_gen"], size=self.nb_decay)

        for key in ["nb_photon", "noise", "irf_length", "irf_shift"]:
            dict_random_parameters[key] = np.random.uniform(params_dict[key]["min_gen"], params_dict[key]["max_gen"],
                                                            size=self.nb_decay)
        is_multi_core = False
        if is_multi_core:
            # def generate_single_exp_decay_fct(tau, t0, noise, nb_photons):
            #     return self.generate_single_exp_decay(tau, t0, noise, nb_photons)

            # FIXME
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
                self.decays.append(self.lifetime_measurement.model.generate(self.time_ns, self.time_idx, dict_random_parameters, i))




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


    def generate_simplest_single_exp_decay(self, tau, t0, noise, nb_of_generated_photon):
        decay = np.exp(-(self.time_ns - t0) / tau)
        decay[self.time_ns < t0] = 0
        decay /= decay.sum()

        decay_obj = rv_discrete(name='mono_exp', values=(self.time_idx, decay))
        photons = decay_obj.rvs(size=nb_of_generated_photon)
        decay_data = np.bincount(photons)
        nb_of_pad_data = self.time_idx.size - decay_data.size
        zeros = np.zeros(nb_of_pad_data)
        decay_data = np.concatenate((decay_data, zeros))

        fitted_params_dict = {}
        fitted_params_dict["amp"] = np.max(decay_data)  # NB : before adding noise

        # Adding noise
        decay_data += np.random.random(self.time_idx.size) * noise
        decay_data = decay_data.astype(np.int)
        aux_params_dict = {}

        fitted_params_dict["tau"] = tau
        fitted_params_dict["t0"] = t0
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
