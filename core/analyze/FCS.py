import multiprocessing as mp

import numpy as np
from lmfit import minimize, Parameters, Model
import matplotlib.pyplot as plt
from lmfit.models import LinearModel, ExponentialModel

# from .correlate import correlate

from threading import Thread

from core.analyze.pycorrelate import pcorrelate, make_loglags

import numba

#https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
# https://sebastianraschka.com/Articles/2014_multiprocessing.html




def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars

from core.analyze.Measurement import Measurements

class OneSpeDiffusion(Model):
    """A exponential decay with a shift in time, with four Parameters ``t0``, ``amp``, ``tau`` and ``cst``.

    Defined as:

    .. math::

        f(t; , G0, tdiff, cst) = cst + G0/(1+t/tdiff)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def oneSpeDiffusion(t, G0, tdiff, cst):
            return cst + G0/(1+t/tdiff)

        super(OneSpeDiffusion, self).__init__(oneSpeDiffusion, **kwargs)

    def guess(self, data, x=None, **kwargs):
        G0, tdiff, cst = 0., 0., 0.
        #if x is not None:
        G0 = data[0] #beware afterpulsing...
        cst = np.mean(data[-10:])
        #Searching for position where G0 is divided by 2
        tdiff = np.argmax(data < (float) (G0) / 2)

        pars = self.make_params(G0=G0, tdiff=tdiff, cst=cst)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"


class CorrelationMeasurement(Measurements):
    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, type="correlation", name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, type, name, comment)

    def correlateMonoProc(self, timestamps1, timestamps2, max_correlation_time_in_tick, start_correlation_time_in_tick=2, nbOfPointPerCascade_aka_B=10, tick_duration_micros=1):
        #FIXME
        timestamps2 = timestamps1

        self.tick_duration_micros = tick_duration_micros
        self.maxTimeInTick = timestamps1[-1]
        self.num_last_photon = np.searchsorted(timestamps1, self.maxTimeInTick - max_correlation_time_in_tick)
        self.end_time_correlation_tick = timestamps1[self.num_last_photon]

        self.create_list_time_correlation(start_correlation_time_in_tick, max_correlation_time_in_tick, pointPerDecade=nbOfPointPerCascade_aka_B)
        # self.data = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)

        # correlate(timestamps, self.data, self.timeAxis, self.numLastPhoton)
        self.data = pcorrelate(t=timestamps1, u=timestamps2, bins=self.timeAxis, normalize=True)
        # self.pcorrelate_me(timestamps1, self.timeAxis, self.data)

        # self.normalize_correlation()
        self.scale_time_axis()
        self.time_axis = self.time_axis[:-1]

    @numba.jit(nopython=True)
    def pcorrelate_me(self, timestamps, taus, G):

        numLastPhoton = np.size(timestamps)
        nbOfTau = np.size(taus)
        for n in range(numLastPhoton):
            # if n%100000==0:
            #     print(n)
            idx_tau = 0
            j = n + 1
            while(idx_tau < nbOfTau):
                # First tau is not necesseraly 1
                while timestamps[j] - timestamps[n]  < taus[0] - 1:
                    j += 1

                while timestamps[j] - timestamps[n]  < taus[idx_tau]:
                    G[idx_tau] += 1
                    j += 1
                    # if j == nbOfPhoton:
                    #     break
                idx_tau += 1

    def correlateFCS_multicore(self, timestamps_1, timestamps_2, max_correlation_time_in_tick, start_correlation_time_in_tick=2, nb_of_point_per_cascade_aka_B=10, tick_duration_micros=1):
        """
        :param timestamps_1:
        :param timestamps_2:
        :param max_correlation_time_in_tick:
        :param start_correlation_time_in_tick:
        :param nb_of_point_per_cascade_aka_B:
        :param tick_duration_micros:
        :return:
        """
        self.tick_duration_micros = tick_duration_micros

        # TODO nb_of_chunk based on max correlation time and the max Time of the file.

        nb_of_chunk = 4
        nb_of_workers = 4

        # Split the timeStamps in 10 (?) array
        self.max_time_in_tick_1 = timestamps_1[-1]
        self.max_time_in_tick_2 = timestamps_2[-1]

        chunks_of_timestamps_1 = np.array_split(timestamps_1, nb_of_chunk)
        chunks_of_timestamps_2 = np.array_split(timestamps_2, nb_of_chunk)

        # TODO fix cross-correlation

        self.num_last_photon = np.searchsorted(timestamps_1, self.max_time_in_tick_1 - max_correlation_time_in_tick)
        self.end_time_correlation_tick = timestamps_1[self.num_last_photon]

        self.create_list_time_correlation(start_correlation_time_in_tick, max_correlation_time_in_tick,
                                          pointPerDecade=nb_of_point_per_cascade_aka_B)

        # make_loglags(exp_min=, exp_max=, points_per_base=nb_of_point_per_cascade_aka_B)
        self.data = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)

        p = mp.Pool(nb_of_workers)
        Gs = [p.apply(pcorrelate, args=(chunks_of_timestamps_1[i], chunks_of_timestamps_2[i], self.time_axis, True)) for i in range(nb_of_chunk)]

        # p.starmap()
        # print(p.map(f, [1, 2, 3]))

        # processes = [mp.Process(target=pcorrelate_b, args=(chunks_of_timestamps_1[i], chunks_of_timestamps_2[i], self.timeAxis, Gs, True)) for i in range(nb_of_workers)]
        # # Run processes
        # for p in processes:
        #     p.start()
        #
        # # Exit the completed processes
        # for p in processes:
        #     p.join()

        Gs = np.vstack(Gs)

        self.data = np.mean(Gs, axis=0)
        self.error_bar = np.std(Gs, axis=0)

        # self.normalize_correlation()
        self.scale_time_axis()
        self.time_axis = self.time_axis[:-1]


    def normalize_correlation(self):
        self.data = self.data.astype(np.float64)
        B = self.pointPerDecade
        for n in range(1, self.nb_of_cascade):
            self.data[n * B:(n + 1) * B] /= np.power(2, n)
        self.data *= (self.maxTimeInTick - self.time_axis) / (self.num_last_photon ** 2)

    def scale_time_axis(self):
        self.time_axis = self.tick_duration_micros * self.time_axis.astype(np.float64)

    def create_list_time_correlation(self, startCorrelationTimeInTick, maxCorrelationTime_tick, pointPerDecade):
        B = self.pointPerDecade = pointPerDecade
        # How many "cascade" do we need ?
        # maxCorrelationTime_tick =  2^(n_casc - 1/B)
        # then ln (maxCorrelationTime_tick) = (n_casc - 1/B) ln 2
        self.nb_of_cascade = int(np.log(maxCorrelationTime_tick) / np.log(2) + 1 / B)

        """
                 |
                 | 1                                si j = 1
         tau_j = |
                 |                  j - 1
                 | tau_(j-1) + 2^( -------)         si j > 1      ATTENTION division entre integer
                                      B                           i.e on prend la partie entiere !!

        """
        # TODO Total vectorisation ?
        self.nbOfCorrelationPoint = int(self.nb_of_cascade * B)  # +1 ?
        taus = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)
        taus[:B] = np.arange(B) + 1
        for n in range(1, self.nb_of_cascade):
            taus[n * B:(n + 1) * B] = taus[:B] * np.power(2, n) + taus[n * B - 1]
        taus += startCorrelationTimeInTick
        self.time_axis = taus

    def set_params(self, params):
        if self.modelName == "1 Diff":
            self.params['G0'].set(value=params[0], vary=True, min=0, max=None)
            self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[2], vary=True, min=0, max=None)

    def set_model(self, modelName):
        # il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "1 Diff":
            self.modelName = modelName
            self.model = OneSpeDiffusion()
            self.params = self.model.make_params(G0=1.5, tdiff=500, cst=1)

    def create_canonic_graph(self, is_plot_error_bar=False, is_plot_text=True):
        self.canonic_fig, self.canonic_fig_ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0)
        ax = self.canonic_fig_ax
        ax[0].semilogx(self.time_axis, self.data, "ro", alpha=0.5)
        for a in ax:
            a.grid(True)
            a.grid(True, which='minor', lw=0.3)

        if self.fit_results is not None:
            ax[0].semilogx(self.time_axis[:-1], self.fit_results.best_fit, "b-", linewidth=3)
        if self.residuals is not None:
            ax[1].semilogx(self.time_axis[:-1], self.fit_results.residual, 'k')
            ym = np.abs(self.fit_results.residual).max()
            ax[1].set_ylim(-ym, ym)
        # ax[1].set_xlim(bins[0]*unit, bins[-1]*unit);
        # tau_diff_us = fitres.values['tau_diff'] * 1e6

        if is_plot_text:
            pass
            # TODO Changer le texte selon les modeles
            if self.modelName == "1 Diff":
                msg = ((
                                   r'$\G_0$ = {G0:.2f} ns' + '\n' + r'$\tau_d$ = {taud:.0f} ns' + '\n' + r'$r$ = {e:.0f}')
                       .format(tau1=self.fit_results.values['G0'], taud=self.fit_results.values['tdiff'], e=0))

            # ax[0].text(.75, .9, msg,
            #            va='top', ha='left', transform=ax[0].transAxes, fontsize=18);

        if is_plot_error_bar and self.error_bar is not None:
            ax[0].semilogx(self.time_axis, self.data + self.error_bar, alpha=0.2)
            ax[0].semilogx(self.time_axis, self.data - self.error_bar, alpha=0.2)
            # ax[0].fill_between(self.time_axis, self.data + self.error_bar, self.data, alpha=0.2)
            # ax[0].fill_between(self.time_axis, self.data - self.error_bar, self.data, alpha=0.2)


        ax[0].set_ylabel('G(τ)', fontsize=40)
        ax[1].set_ylabel('residuals', fontsize=20)
        ax[1].set_xlabel('Time Lag, τ (µs)', fontsize=40)

        ax[0].tick_params(axis='both', which='major', labelsize=20)
        ax[0].tick_params(axis='both', which='minor', labelsize=8)

        ax[1].tick_params(axis='x', which='major', labelsize=20)
        ax[1].tick_params(axis='x', which='minor', labelsize=8)

        return self.canonic_fig


class FCSMeasurements(CorrelationMeasurement):

    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "FCS", name, comment)

    def set_params(self, params):
        if self.modelName == "1 Diff":
            self.params['G0'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[2], vary=True, min=0, max=None)

    def set_model(self, modelName):
        #il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "1 Diff":
            self.modelName = modelName
            self.model = OneSpeDiffusion()
            self.params = self.model.make_params(G0=1.5, tdiff=500, cst=1)




