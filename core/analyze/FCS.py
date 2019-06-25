import multiprocessing as mp

import numpy as np
from lmfit import minimize, Parameters, Model
import matplotlib.pyplot as plt
from lmfit.models import LinearModel, ExponentialModel

from .correlate import whal_auto

from threading import Thread

from core.analyze.pycorrelate import pcorrelate, make_loglags, ucorrelate_coeff, pnormalize, find_pair_Whal

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
    """A correlation curve for a diffusion inside a gaussiean MDF, with four Parameters ``G0``, ``tdiff``, ``r`` and ``cst``.

    Defined as:

    .. math::

        f(t; , G0, tdiff, cst) = cst + G0/(1+t/tdiff)*1/sqrt(1+r*t/tdiff)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def oneSpeDiffusion(t, G0, tdiff, r, cst):
            return cst + G0/(1+t/tdiff)*1/np.sqrt(1+r*t/tdiff)

        super(OneSpeDiffusion, self).__init__(oneSpeDiffusion, **kwargs)

    def guess(self, data, x=None, **kwargs):
        G0, tdiff, r, cst = 0., 0., 0., 0.
        #if x is not None:
        G0 = data[0] - 1 #beware afterpulsing...
        cst = np.mean(data[-10:])
        #Searching for position where G0 is divided by 2
        tdiff = np.argmax(data < (float) (G0)/2 + cst)
        # FIXME
        r = 1

        pars = self.make_params(G0=G0, tdiff=tdiff, r=r, cst=cst)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"

class TwoSpeDiffusion(Model):
    """A exponential decay with a shift in time, with four Parameters ``t0``, ``amp``, ``tau`` and ``cst``.

    Defined as:

    .. math::

        f(t; , G0, tdiff, cst) = cst + G0a/(1+t/tdiffa) + G0b/(1+t/tdiffb)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def twoSpeDiffusion(t, G0a, tdiffa, cst, G0b, tdiffb):
            return cst + G0a/(1+t/tdiffa) + G0b/(1+t/tdiffb)

        super(TwoSpeDiffusion, self).__init__(twoSpeDiffusion, **kwargs)

    def guess(self, data, x=None, **kwargs):
        # FIXME
        G0a, tdiffa, G0b, tdiffb, cst = 0., 0., 0.
        #if x is not None:
        G0a = data[0] - 1 #beware afterpulsing...
        cst = np.mean(data[-10:])
        #Searching for position where G0 is divided by 2
        tdiffa = np.argmax(data < (float) (G0a)/2 + cst)

        pars = self.make_params(G0a=G0a, tdiffa=tdiffa, cst=cst, G0b=G0a, tdiffb=tdiffa)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"


class CorrelationMeasurement(Measurements):
    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, type="correlation", name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, type, name, comment)
        self.num_c1 = 0
        self.num_c2 = 0
        self.start_cor_time_micros = 10
        self.max_cor_time_ms = 1000
        self.precision = 10

    def correlateMonoProc(self, timestamps_1, timestamps_2, max_correlation_time_in_tick, start_correlation_time_in_tick=2, nb_of_point_per_cascade_aka_B=10, tick_duration_micros=1):
        #FIXME
        timestamps_2 = timestamps_1

        self.tick_duration_micros = tick_duration_micros
        self.maxTimeInTick = timestamps_1[-1]
        self.num_last_photon = np.searchsorted(timestamps_1, self.maxTimeInTick - max_correlation_time_in_tick)
        self.end_time_correlation_tick = timestamps_1[self.num_last_photon]

        self.create_list_time_correlation(start_correlation_time_in_tick, max_correlation_time_in_tick, point_per_decade=nb_of_point_per_cascade_aka_B)
        # self.data = np.zeros(self.nbOfCorrelationPoint, dtype=np.int)

        # correlate(timestamps, self.data, self.timeAxis, self.numLastPhoton)


        coeff = np.ones(timestamps_1.size, dtype=np.uint32)
        # timestamps_1_cpy = np.copy(timestamps_1)
        # G = np.zeros(self.time_axis.size, dtype=np.uint32)


        # self.data = ucorrelate_coeff(t_stamps_a=timestamps_1, coeff=coeff, max_lag=max_correlation_time_in_tick)
        # G = whal_auto(timestamps_1_cpy, coeff, self.time_axis, G, B=10)

        # Lawrence algo
        self.data = pcorrelate(t=timestamps_1, u=timestamps_2, bins=self.time_axis, normalize=True)

        # Whal algo
        # G = self.correlate_whal(timestamps_1, coeff, timestamps_2, coeff, self.time_axis, B=10)
        # self.data = G
        # self.normalize_correlation()


        # self.data = pnormalize(G, timestamps_1, timestamps_1, self.time_axis)
        # plt.semilogx(self.data)
        # plt.show()
        # self.data = self.unifom_lag_to_binned_lag(self.data)
        # self.normalize_correlation()

        # self.pcorrelate_me(timestamps1, self.timeAxis, self.data)

        self.scale_time_axis()
        self.time_axis = self.time_axis[:-1]



    def correlate_whal(self, t_stamp_a, coeff_a, t_stamp_b, coeff_b, lags, B=10):
        """
        lags in tick
        :param t_stamps_a:
        :param lags:
        :return:
        """
        G = np.zeros(lags.size)
        if id(t_stamp_a) == id(t_stamp_b):
            is_auto_cor= True
        else:
            is_auto_cor = False

        t_stamp_a = np.copy(t_stamp_a)
        if is_auto_cor:
            t_stamp_b = t_stamp_a
        else:
            t_stamp_b = np.copy(t_stamp_b)


        coaserning_value = 1
        coarsening_counter = 0
        def coarsen_timeStamp(t_stamps, coeffs):
            # NB : //= -> integer division
            t_stamps //= 2
            # find the position of consecutive idx with same timestamp (i.e difference = 0)
            consecutive_idxs = np.argwhere(np.diff(t_stamps) == 0)
            # Merge weighting (coeff) of same timestamps
            coeffs[consecutive_idxs] += coeffs[consecutive_idxs + 1]
            # Removing duplicate timestamps

            t_stamp_a = np.delete(t_stamps, consecutive_idxs + 1)
            coeff_a = np.delete(coeffs, consecutive_idxs + 1)


            # idx_to_keep = np.nonzero(np.diff(t_stamp_a))
            # # idx_to_keep = np.invert(consecutive_idxs)
            # t_stamp_a = t_stamp_a[idx_to_keep]
            # coeff_a = coeff_a[idx_to_keep]

        for idx_G, lag in enumerate(lags):
            if coarsening_counter == B:
                coarsen_timeStamp(t_stamp_a, coeff_a)
                coaserning_value *= 2
                if not is_auto_cor:
                    coarsen_timeStamp(t_stamp_b, coeff_b)
                else:
                    t_stamp_b = t_stamp_a


                coarsening_counter = 0
            else:
                coarsening_counter += 1

            # Pair calculation


            # Lag as also to be divided by 2 for each cascade
            corrected_lag = int(lag / np.power(2, idx_G//B))

            # Short numpy implementation that is quite slow bevause it does'nt take into account the fact that the list are ordered.
            # correlation_match = np.in1d(t_stamp_a, t_stamp_a_dl, assume_unique=True)
            # G[idx_G] = np.sum(coeff_a[correlation_match])

            # Numba or Cython implementation
            G[idx_G] = find_pair_Whal(t_stamp_a, coeff_a, t_stamp_b, coeff_b, corrected_lag)
            G[idx_G] /= coaserning_value

        return G

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
                while timestamps[j] - timestamps[n] < taus[0] - 1:
                    j += 1

                while timestamps[j] - timestamps[n] < taus[idx_tau]:
                    G[idx_tau] += 1
                    j += 1
                    # if j == nbOfPhoton:
                    #     break
                idx_tau += 1

    def unifom_lag_to_binned_lag(self, G_u):
        lags = self.time_axis
        G = np.zeros(lags.size)
        for i in range(lags.size):
            G[i] = np.sum(G_u[lags[i]:lags[i+1]])
        return G

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
                                          point_per_decade=nb_of_point_per_cascade_aka_B)

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
        # self.time_axis = self.time_axis[:-1]


    def normalize_correlation(self):
        self.data = self.data.astype(np.float64)
        B = self.point_per_decade
        for n in range(1, self.nb_of_cascade):
            self.data[n * B:(n + 1) * B] /= np.power(2, n)
        self.data *= (self.maxTimeInTick - self.time_axis) / (self.num_last_photon ** 2)

    def scale_time_axis(self):
        self.time_axis = self.tick_duration_micros * self.time_axis.astype(np.float64)

    def create_list_time_correlation(self, start_correlation_time_in_tick, max_correlation_time_tick, point_per_decade):
        B = self.point_per_decade = point_per_decade
        # How many "cascade" do we need ?
        # maxCorrelationTime_tick =  2^(n_casc - 1/B)
        # then ln (maxCorrelationTime_tick) = (n_casc - 1/B) ln 2
        self.nb_of_cascade = int(np.log(max_correlation_time_tick) / np.log(2) + 1 / B)

        """
                 |
                 | 1                                si j = 1
         tau_j = |
                 |                  j - 1
                 | tau_(j-1) + 2^( -------)         si j > 1      ATTENTION division entre integer
                                      B                           i.e on prend la partie entiere !!

        """
        # TODO Total vectorisation ? Numba ?
        self.nbOfCorrelationPoint = int(self.nb_of_cascade * B)  # +1 ?
        taus = np.zeros(self.nbOfCorrelationPoint, dtype=np.uint32)
        taus[:B] = np.arange(B) + 1
        for n in range(1, self.nb_of_cascade):
            taus[n * B:(n + 1) * B] = taus[:B] * np.power(2, n) + taus[n * B - 1]
        taus += start_correlation_time_in_tick
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

    def create_canonic_graph(self, is_plot_error_bar=False, is_plot_text=True, save_file_name=""):
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

        if save_file_name != "":
            plt.savefig(save_file_name, dpi=300)

        return self.canonic_fig


class FCSMeasurements(CorrelationMeasurement):

    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "FCS", name, comment)


    def remove_afterpulsing_via_FLCS(self, nanotimes):
        decay = np.bincount(nanotimes)

        P = np.zeros((2,self.exp_param.nb_of_microtime_channel))
        P[0,:] = decay / decay.sum()
        P[1,:] = 1.0/self.exp_param.nb_of_microtime_channel
        Pt = np.transpose(P)

        I = np.diag(decay)
        inverse_I = np.linalg.inv(I)

        P_InvI_Pt = np.dot(P, np.dot(inverse_I, Pt))

        Inv_P_InvI_Pt = np.linalg.inv(P_InvI_Pt)

        f = np.dot(Inv_P_InvI_Pt, np.dot(P, inverse_I))

        self.filter_after_pulsing_coeff = f[0,:]


    def set_params(self, params):
        if self.modelName == "1 Diff":
            self.params['G0'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
            self.params['r'].set(value=params[2], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[3], vary=True, min=0, max=None)
        # if self.modelName == "2 Diff":
        #     self.params['G0a'].set(value=params[0],  vary=True, min=0, max=None)
        #     self.params['tdiffa'].set(value=params[1], vary=True, min=0, max=None)
        #     self.params['G0b'].set(value=params[0],  vary=True, min=0, max=None)
        #     self.params['tdiffb'].set(value=params[1], vary=True, min=0, max=None)
        #     self.params['cst'].set(value=params[2], vary=True, min=0, max=None)

    def set_model(self, modelName):
        #il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "1 Diff":
            self.modelName = modelName
            self.model = OneSpeDiffusion()
            self.params = self.model.make_params(G0=1.5, tdiff=500, r=10, cst=1)
        if modelName == "2 Diff":
            self.modelName = modelName
            self.model = TwoSpeDiffusion()
            self.params = self.model.make_params(G0a=1.5, tdiffa=500, cst=1, G0b=1.5, tdiffb=500)




