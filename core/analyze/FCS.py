import multiprocessing as mp

import numpy as np
from lmfit import minimize, Parameters, Model
import matplotlib.pyplot as plt
from lmfit.models import LinearModel, ExponentialModel

from .correlate import whal_auto

from threading import Thread

from core.analyze.pycorrelate import pcorrelate, make_loglags, ucorrelate_coeff, pnormalize, find_pair_Whal, pnormalize_coeff, coarsen_timestamp, pnormalize_coeff_whal, correlate_whal, find_correlation_area

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

        f(t; , G0, tdiff, cst) = cst + G0a/(1+t/tdiffa)*1/sqrt(1+r*t/tdiffa) + G0b/(1+t/tdiffb)*1/sqrt(1+r*t/tdiffb)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def twoSpeDiffusion(t, G0a, tdiffa, r, cst, G0b, tdiffb):
            return cst + G0a/(1+t/tdiffa)*1/np.sqrt(1+r*t/tdiffa) + G0b/(1+t/tdiffb)*1/np.sqrt(1+r*t/tdiffb)

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
    def __init__(self, exps, exp, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, type="correlation", name="", comment=""):
        super().__init__(exps, exp, exp_param, num_channel, start_tick, end_tick, type, name, comment)
        self.num_c1 = 0
        self.num_c2 = 0
        self.start_cor_time_micros = 10
        self.max_cor_time_ms = 1000
        self.precision = 10
        self.sub_correlation_curves = None
        self.is_multi_proc = True
        self.algo = "Whal"

    def set_additional_param_for_calculation(self, params):
        self.num_c1, self.num_c2, self.start_cor_time_micros, self.max_cor_time_ms, self.is_multi_proc, self.precision, self.algo = params

    def calculate(self):
        start_tick = self.start_tick
        end_tick = self.end_tick

        if start_tick == 0:
            start_tick = self.exp.data.channels[self.num_c1].start_tick

        if end_tick == -1:
            end_tick = self.exp.data.channels[self.num_c1].end_tick

        timeStamps_1 = self.exp.data.channels[self.num_c1].photons['timestamps']
        idxStart, idxEnd = np.searchsorted(timeStamps_1, (start_tick, end_tick))
        timeStamps_1_reduc = timeStamps_1[idxStart:idxEnd]

        timeStamps_2 = self.exp.data.channels[self.num_c2].photons['timestamps']
        idxStart, idxEnd = np.searchsorted(timeStamps_2, (start_tick, end_tick))
        timeStamps_2_reduc = timeStamps_2[idxStart:idxEnd]

        max_correlation_time_in_tick = int(
            self.max_cor_time_ms / 1E3 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        start_correlation_time_in_tick = int(
            self.start_cor_time_micros / 1E6 / self.exp_param.mAcrotime_clickEquivalentIn_second)

        tick_duration_micros = self.exp_param.mAcrotime_clickEquivalentIn_second*1E6

        #FIXME from afterpulsing filter
        coeff_1 = np.ones(timeStamps_1_reduc.size, dtype=np.uint32)
        coeff_2 = np.ones(timeStamps_2_reduc.size, dtype=np.uint32)

        if self.is_multi_proc:
            fct = self.correlate_multicore
        else:
            fct = self.correlate_mono_proc

        fct(timeStamps_1_reduc, timeStamps_2_reduc, coeff_1, coeff_2,
                                            max_correlation_time_in_tick, start_correlation_time_in_tick, self.precision, tick_duration_micros, algo=self.algo)

        return self

    def correlate_mono_proc(self, timestamps_1, timestamps_2, coeff_1, coeff_2, max_correlation_time_in_tick, start_correlation_time_in_tick=2, nb_of_point_per_cascade_aka_B=10, tick_duration_micros=1, algo="Whal"):

        # if id(t_stamp_a) == id(t_stamp_b):
        if np.array_equal(timestamps_1, timestamps_2):
            is_auto_cor = True
            timestamps_2 = timestamps_1
        else:
            is_auto_cor = False

        self.error_bar = None

        self.tick_duration_micros = tick_duration_micros
        self.maxTimeInTick = timestamps_1[-1]
        self.num_last_photon = np.searchsorted(timestamps_1, self.maxTimeInTick - max_correlation_time_in_tick)
        self.end_time_correlation_tick = timestamps_1[self.num_last_photon]

        self.create_list_time_correlation(start_correlation_time_in_tick, max_correlation_time_in_tick, point_per_decade=nb_of_point_per_cascade_aka_B)

        if algo == "Whal":
            self.data = correlate_whal(timestamps_1, coeff_1, timestamps_2, coeff_2, lags=self.time_axis, B=10, is_normalize=True, is_auto_cor=is_auto_cor)
            # is_normalize = True
            # if is_normalize:
            #     self.data = pnormalize_coeff_whal(G, timestamps_1, timestamps_2, bins=self.time_axis)
            self.scale_time_axis()
        elif algo == "Laurence":
            self.data = pcorrelate(t=timestamps_1, u=timestamps_2, bins=self.time_axis, normalize=True)
            self.scale_time_axis()
            self.time_axis = self.time_axis[:-1]
        # self.pcorrelate_me(timestamps1, self.timeAxis, self.data)


    def pcorrelate_me(self, timestamps, taus, G):
        numLastPhoton = np.size(timestamps)
        nbOfTau = np.size(taus)
        for n in range(numLastPhoton):
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

    def correlate_multicore(self, timestamps_1, timestamps_2, coeff_1, coeff_2, max_correlation_time_in_tick, start_correlation_time_in_tick=2, nb_of_point_per_cascade_aka_B=10, tick_duration_micros=1, algo="Whal"):
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
        # TODO nb_of_workers based on user preference.
        nb_of_workers = 4

        # Split the timeStamps in array
        self.max_time_in_tick_1 = timestamps_1[-1]
        self.max_time_in_tick_2 = timestamps_2[-1]

        # Arbitrary statement : In order to compute lag at time time, we need at least ratio_lag_vs_file_duration time
        # of data

        # TODO minimum number of photon in a chunk ?
        chunk_boundaries = []
        ratio_lag_vs_file_duration = 6

        # nb_of_chunk = int(self.max_time_in_tick_1/(max_correlation_time_in_tick*ratio_lag_vs_file_duration))

        nb_of_max_cor_time = self.max_time_in_tick_1/max_correlation_time_in_tick

        i = 0
        N = timestamps_1.size
        chunk_time_limit = i * ratio_lag_vs_file_duration * max_correlation_time_in_tick
        boundary = np.searchsorted(timestamps_1, chunk_time_limit)
        while boundary != N:
            chunk_boundaries.append(boundary)
            i += 1
            chunk_time_limit = i * ratio_lag_vs_file_duration * max_correlation_time_in_tick
            boundary = np.searchsorted(timestamps_1, chunk_time_limit)

        nb_of_chunk = len(chunk_boundaries)
        if nb_of_chunk > 15:
            nb_of_chunk = 15

        self.log("nb of chunk : %d\n" % nb_of_chunk)

        # FIXME set minimum value for nb_of_chunk
        # if nb_of_chunk > 10:
        #     nb_of_chunk = 10
        # elif nb_of_chunk < 10:
        #     # diminish the max correlation_time ?
        #     pass

        chunks_of_timestamps_1 = []
        chunks_of_timestamps_2 = []
        chunks_of_coeff_1 = []
        chunks_of_coeff_2 = []

        for i in range(len(chunk_boundaries) - 1):
            b1_1 = chunk_boundaries[i]
            b2_1 = chunk_boundaries[i+1]
            chunks_of_timestamps_1.append(timestamps_1[b1_1:b2_1])
            chunks_of_coeff_1.append(coeff_1[b1_1:b2_1])
            b1_2 = min(b1_1, timestamps_2.size-1)
            b2_2 = min(b2_1, timestamps_2.size-1)
            chunks_of_timestamps_2.append(timestamps_2[b1_2:b2_2])
            chunks_of_coeff_2.append(coeff_2[b1_2:b2_2])



        # chunks_of_timestamps_1 = np.array_split(timestamps_1, nb_of_chunk)
        # chunks_of_timestamps_2 = np.array_split(timestamps_2, nb_of_chunk)
        # chunks_of_coeff_1 = np.array_split(coeff_1, nb_of_chunk)
        # chunks_of_coeff_2 = np.array_split(coeff_2, nb_of_chunk)

        # TODO fix cross-correlation

        self.num_last_photon = np.searchsorted(timestamps_1, self.max_time_in_tick_1 - max_correlation_time_in_tick)
        self.end_time_correlation_tick = timestamps_1[self.num_last_photon]

        self.time_axis = self.create_list_time_correlation(start_correlation_time_in_tick, max_correlation_time_in_tick,
                                          point_per_decade=nb_of_point_per_cascade_aka_B)

        self.data = np.zeros(self.nb_of_correlation_point, dtype=np.int)

        self.log("Creating a pool of %d workers\n" % nb_of_workers)
        p = mp.Pool(nb_of_workers)
        self.log("Calculating Correlation \n")
        if algo == "Laurence":
            # Gs = [p.apply(pcorrelate, args=(chunks_of_timestamps_1[i], chunks_of_timestamps_2[i], self.time_axis, True)) for i in range(nb_of_chunk-1)]
            results = [p.apply_async(pcorrelate, args=(chunks_of_timestamps_1[i], chunks_of_timestamps_2[i], self.time_axis, True)) for i in range(nb_of_chunk-1)]
            Gs = [p.get() for p in results]
        elif algo == "Whal":
            # Gs = [p.apply(correlate_whal,
            #               args=(chunks_of_timestamps_1[i], chunks_of_coeff_1[i], chunks_of_timestamps_2[i], chunks_of_coeff_2[i], self.time_axis, 10, True, True)) for i
            #       in range(nb_of_chunk - 1)]
            results = [p.apply_async(correlate_whal,
                          args=(chunks_of_timestamps_1[i], chunks_of_coeff_1[i], chunks_of_timestamps_2[i], chunks_of_coeff_2[i], self.time_axis, 10, True, True)) for i
                  in range(nb_of_chunk - 1)]
            Gs = [p.get() for p in results]

        elif algo == "F2Cor":
            pass
        elif algo == "lin":
            pass
        self.sub_correlation_curves = np.vstack(Gs)

        self.data = np.mean(self.sub_correlation_curves, axis=0)
        self.error_bar = np.std(self.sub_correlation_curves, axis=0)

        # self.normalize_correlation()
        self.scale_time_axis()

        if algo == "Laurence":
            self.time_axis = self.time_axis[:-1]
        self.log("Calculation complete !\n")

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
        self.nb_of_correlation_point = int(self.nb_of_cascade * B)  # +1 ?
        taus = np.zeros(self.nb_of_correlation_point, dtype=np.uint32)
        taus[:B] = np.arange(B) + 1
        for n in range(1, self.nb_of_cascade):
            taus[n * B:(n + 1) * B] = taus[:B] * np.power(2, n) + taus[n * B - 1]
        i = -1
        while taus[i] > max_correlation_time_tick:
            taus[i] = 0
            i -= 1
        taus = np.trim_zeros(taus)
        taus += start_correlation_time_in_tick
        self.nb_of_correlation_point = taus.size
        self.time_axis = taus
        return taus

    def create_chronogram_overlay(self, chronogram, correl_tick1, correl_tick2):
        if correl_tick1 > correl_tick2:
            correl_tick1, correl_tick2 = correl_tick2, correl_tick1

        correl_tick1 = int(correl_tick1 / (self.exp_param.mAcrotime_clickEquivalentIn_second*1E6))
        correl_tick2 = int(correl_tick2 / (self.exp_param.mAcrotime_clickEquivalentIn_second*1E6))


        timestamps = chronogram.get_raw_data(type="timestamp", mode="data")
        # TODO Extend on left and right ?
        # tick_left = chronogram.start_tick - correl_tick2
        # tick_right = chronogram.end_tick + correl_tick2
        #
        # timestamps = chronogram.get_raw_data(type="timestamp", mode="full")
        # idx_l, idx_r = np.searchsorted(timestamps, (tick_left, tick_right))
        # timestamps = timestamps[idx_l:idx_r]

        # TODO multiproc if too much photons
        # TODO cross correlation
        self.correl_scores = find_correlation_area(timestamps, timestamps, correl_tick1, correl_tick2)

        num_photon_edge = chronogram.get_num_photon_of_edge(is_absolute_number=False)

        multi_dim_mask = np.hsplit(self.correl_scores, num_photon_edge)
        # multi_dim_selected_photon_by_bin = np.cumsum(multi_dim_mask, axis=1)
        correl_score_by_bin = np.zeros(len(multi_dim_mask))
        for i, dim in enumerate(multi_dim_mask):
            correl_score_by_bin[i] = np.sum(dim)

        return correl_score_by_bin




    # def set_params(self, params):
    #     if self.modelName == "1 Diff":
    #         self.params['G0'].set(value=params[0], vary=True, min=0, max=None)
    #         self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
    #         self.params['cst'].set(value=params[2], vary=True, min=0, max=None)

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

    def __init__(self, exps, exp, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exps, exp, exp_param, num_channel, start_tick, end_tick, "FCS", name, comment)


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


    # def set_params(self, params):
    #     if self.modelName == "1 Diff":
    #         self.params['G0'].set(value=params[0],  vary=True, min=0, max=None)
    #         self.params['tdiff'].set(value=params[1], vary=True, min=0, max=None)
    #         self.params['r'].set(value=params[2], vary=True, min=0, max=None)
    #         self.params['cst'].set(value=params[3], vary=True, min=0, max=None)
    #     if self.modelName == "2 Diff":
    #         self.params['G0a'].set(value=params[0],  vary=True, min=0, max=None)
    #         self.params['tdiffa'].set(value=params[1], vary=True, min=0, max=None)
    #         self.params['r'].set(value=params[2], vary=True, min=0, max=None)
    #         self.params['cst'].set(value=params[3], vary=True, min=0, max=None)
    #         self.params['G0b'].set(value=params[4],  vary=True, min=0, max=None)
    #         self.params['tdiffb'].set(value=params[5], vary=True, min=0, max=None)


    def set_model(self, modelName):
        #il existe une  possibilité pour autoriser le passage d’une infinité de paramètres ! Cela se fait avec *
        if modelName == "1 Diff":
            self.modelName = modelName
            self.model = OneSpeDiffusion()
            self.params = self.model.make_params(G0=1.5, tdiff=500, r=1.2, cst=1)
        if modelName == "2 Diff":
            self.modelName = modelName
            self.model = TwoSpeDiffusion()
            self.params = self.model.make_params(G0a=1.5, tdiffa=500, r=1.2, cst=1, G0b=1.5, tdiffb=500)




