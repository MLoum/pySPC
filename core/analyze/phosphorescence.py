import numpy as np
from lmfit import minimize, Parameters, Model, fit_report, Minimizer
from lmfit.models import LinearModel, ExponentialModel
import matplotlib.pyplot as plt



from scipy.special import erfc
from scipy.ndimage.interpolation import shift as shift_scipy

from .Measurement import Measurements
from .histogram import histogram

import numba

def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars

class OneExpDecay(Model):
    """representing the convolution of a gaussian and an exponential. This function is used to model the IRF.

    Defined as:

    .. math::

        f(x;\mu,\sigma,\lambda) = \frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):

        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def oneExpDecay(t, t0, amp, tau, cst):
            exp_decay = cst + amp * np.exp(-(t - t0) / tau)
            exp_decay[t < t0] = cst
            return exp_decay


        super(OneExpDecay, self).__init__(oneExpDecay, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pass
        # return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"


@numba.jit(nopython=True, nogil=True)
def calculate_delay(timestamps_start, timestamps_stop):
    cpt_stop = 0
    max_cpt_stop = len(timestamps_stop)
    cpt_delays = 0
    delays = np.zeros(timestamps_stop.size)
    for cpt_start in range(len(timestamps_start) - 1):
        while cpt_stop < max_cpt_stop and timestamps_stop[cpt_stop] < timestamps_start[cpt_start + 1]:
            delays[cpt_delays] = timestamps_stop[cpt_stop] - timestamps_start[cpt_start]
            cpt_delays += 1
            cpt_stop += 1
    return delays


class PhosphoMeasurements(Measurements):

    def __init__(self, exps, exp, exp_param=None, num_channel=0,  start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exps, exp, exp_param, num_channel, start_tick, end_tick, "phosphorescence", name, comment)
        self.num_start = 0
        self.num_stop = 1
        self.time_step_micros = 10
        self.max_time_histo_ms = 5000
        self.min_time_histo_micros = 10

    def calculate(self):

        # We need two start and end tick, one for the start detector and one for the stop detector.
        start_tick_start = start_tick_stop = self.start_tick
        end_tick_start = end_tick_stop = self.end_tick

        timestamps_start = self.exp.data.channels[self.num_start].photons['timestamps']
        timestamps_stop = self.exp.data.channels[self.num_stop].photons['timestamps']
        if start_tick_start == 0:
            start_tick_start = self.exp.data.channels[self.num_start].start_tick
            start_tick_stop = self.exp.data.channels[self.num_stop].start_tick

        if end_tick_start == -1:
            end_tick_start = self.exp.data.channels[self.num_stop].end_tick
            end_tick_stop = self.exp.data.channels[self.num_stop].end_tick

        idxStart, idxEnd = np.searchsorted(timestamps_start, (start_tick_start, end_tick_start))
        timestamps_start = timestamps_start[idxStart:idxEnd]
        idxStart, idxEnd = np.searchsorted(timestamps_stop, (start_tick_stop, end_tick_stop))
        timestamps_stop = timestamps_stop[idxStart:idxEnd]

        diff_tick_start = np.diff(timestamps_start)

        self.create_histogramm(timestamps_start, timestamps_stop, self.time_step_micros)
        return self

    def set_additional_param_for_calculation(self, params):
        self.num_start, self.num_stop, self.time_step_micros, self.min_time_histo_micros, self.max_time_histo_ms = params
        pass

    def create_histogramm(self, timestamps_start, timestamps_stop, time_step_micros):
        """
        :param num_det_start: Number of the start detector
        :param num_det_stop: Number of the stop detector
        :param time_step_micros: Binnin time step in microsecond for the decay curve
        :return:
        """
        pass

        # TODO check if the timestamps_start are valid, i.e. a list of single tick separated by a long (~1s) and quite stable  time period


        # The data in timestamps_start are a succession of experiment (i.e. one laser pulse and then one phosphorescence decay).
        # The starting time of each experiment are recorded in timestamps_start
        # The first step is to separate the data in timestamps_stop in single experiment sub array.

        # Note : Il faudrait s'assurer que les deux expériences sont sur la "même horloge".
        timestamps_start = np.copy(timestamps_start)
        # Test pour verifier si les ticks start sont bien equirepartis.
        delay_start = np.diff(timestamps_start)
        idx_strange_delay = np.abs(delay_start - delay_start.mean()) > 3*delay_start.std()
        strange_delay = delay_start[idx_strange_delay]

        """
        timestamps_stop = np.copy(timestamps_stop)
        idx_start_exp = np.searchsorted(timestamps_stop, timestamps_start, side='left')

        # Nbre de photon par expériences.
        mean_nb_photon_per_experiment = np.mean(np.diff(idx_start_exp))
        std_dev_nb_photon_per_experiment = np.std(np.diff(idx_start_exp))
        """

        """
        # Nouvel Algo
        
        # On supprime les photons du detecteur stop qui sont arrtivés avant le premier start, on n'a pas de référence
        # temporelle pour les placer sur l'histo de phospho
        timestamps_stop = timestamps_stop[idx_start_exp[0]:]
        idx_start_exp -= idx_start_exp[0]


        for i in range(timestamps_start.size - 1):
            # if timestamps_start[i] > timestamps_stop[idx_start_exp[i]]:
            #     dummy = 1
            timestamps_stop[idx_start_exp[i]:idx_start_exp[i+1]] -= timestamps_start[i]

        binning_factor = int(time_step_micros*1E-6/self.exp_param.mAcrotime_clickEquivalentIn_second)
        if binning_factor == 0:
            binning_factor = 1
        # single_exp_array = [single_exp_array / binning_factor for x in single_exp_array ]
        timestamps_stop = (timestamps_stop//binning_factor).astype(np.int64)


        max_delay = (self.max_time_histo_ms/1000 / self.exp_param.mAcrotime_clickEquivalentIn_second)//binning_factor
        min_delay = (self.min_time_histo_micros/1E6/ self.exp_param.mAcrotime_clickEquivalentIn_second)//binning_factor
        idx_to_keep = (timestamps_stop < max_delay)
        timestamps_stop = timestamps_stop[idx_to_keep]
        idx_to_keep = timestamps_stop > min_delay
        timestamps_stop = timestamps_stop[idx_to_keep]

        # idx_long_delay = timestamps_stop > 1000
        # long_timestop = timestamps_stop[idx_long_delay]
        # np.savetxt("C:\openCV\debug.txt", timestamps_stop)

        self.data = np.bincount(timestamps_stop)
        self.time_axis = np.arange(0, self.data.size) * binning_factor * self.exp_param.mAcrotime_clickEquivalentIn_second * 1E6   # in µs
        """

        """
        # Ancien Algo
        single_exp_array = np.split(timestamps_stop, idx_start_exp)

        # we discard the first experiment since it very improbable that the measurement started on a start tick. Consequently has no start and is  incomplete.
        single_exp_array = single_exp_array[1:]



        # For each single experiment, we reset the starting time at zero by removing the time offset timestamps_start[i]
        i = 0
        for exp in range(len(single_exp_array)-1):  #-1 because we have removed the first one.
            exp -= timestamps_start[i]
            i += 1

        # The macro time clock is around 25ns which is very very short compared to a phosophorescence decay (1ms - 10s)
        # Consequently we have to bin the data
        binning_factor = int(time_step_micros*1E-6/self.exp_param.mAcrotime_clickEquivalentIn_second)
        if binning_factor == 0:
            binning_factor = 1
        # single_exp_array = [single_exp_array / binning_factor for x in single_exp_array ]

        for exp in single_exp_array:
            exp //= binning_factor

        single_exp_array_concatenated = np.concatenate(single_exp_array)

        # Then, we create the phosphorescence decay curve by adding all the sub experiment curve. We need to know


        # # Remove empty experiment if any
        # single_exp_array = [x for x in single_exp_array if x.size != 0]
        #
        # max_time_bin = 0
        # for exp in single_exp_array:
        #     if exp[-1] > max_time_bin:
        #         max_time_bin = exp[-1]

        
        # Finally, we compute
        single_exp_array_concatenated = np.asarray(single_exp_array_concatenated, dtype=np.int64)
        max_delay = np.max(single_exp_array_concatenated)
        min_delay = np.min(single_exp_array_concatenated)
        self.data = np.bincount(single_exp_array_concatenated)
        nb_photon = np.sum(self.data)
        self.time_axis = np.arange(0, self.data.size) * binning_factor * self.exp_param.mAcrotime_clickEquivalentIn_second * 1E6   # in µs
        idx_start = np.searchsorted(self.time_axis, self.min_time_histo_micros)
        idx_end = np.searchsorted(self.time_axis, self.max_time_histo_ms*1000)
        self.time_axis = self.time_axis[idx_start:idx_end]
        self.data = self.data[idx_start:idx_end]
        """

        idx_first_start_in_stop = np.searchsorted(timestamps_stop, timestamps_start[0])
        timestamps_stop = timestamps_stop[idx_first_start_in_stop:]

        """
        cpt_start = 0
        cpt_stop = 0
        cpt_delays = 0
        delays = np.zeros(timestamps_stop.size)
        for cpt_start in range(len(timestamps_start) - 1):
            while timestamps_stop[cpt_stop] < timestamps_start[cpt_start + 1]:
                delays[cpt_delays] = timestamps_stop[cpt_stop] - timestamps_start[cpt_start]
                cpt_delays += 1
                cpt_stop += 1
        """
        #TODO vectorize (numba is fine ?)
        delays = calculate_delay(timestamps_start, timestamps_stop)

        # The macro time clock is around 25ns which is very very short compared to a phosophorescence decay (1ms - 10s)
        # Consequently we have to bin the data
        binning_factor = int(time_step_micros*1E-6/self.exp_param.mAcrotime_clickEquivalentIn_second)
        if binning_factor == 0:
            binning_factor = 1

        delays //= binning_factor

        delays = np.asarray(delays, dtype=np.int64)
        self.data = np.bincount(delays)

        nb_photon = np.sum(self.data)
        self.time_axis = np.arange(0, self.data.size) * binning_factor * self.exp_param.mAcrotime_clickEquivalentIn_second * 1E6   # in µs
        idx_start = np.searchsorted(self.time_axis, self.min_time_histo_micros)
        idx_end = np.searchsorted(self.time_axis, self.max_time_histo_ms*1000)
        self.time_axis = self.time_axis[idx_start:idx_end]
        self.data = self.data[idx_start:idx_end]



    def create_canonic_graph(self, is_plot_error_bar=False, is_plot_text=False):
        pass

    def set_model(self, model_name):

        if model_name == "One Decay Tail":
            self.modelName = model_name
            self.model = OneExpDecay()
            self.params = self.model.make_params(t0=0, amp=1, tau=1, cst=0)
            self.model.fit_formula = r"f(t; amp, \tau, t0, cst) = amp (e^{-(t-t0)/\tau}) + cst"

        # elif model_name == "Two Decays Tail":
        #     # print (modelName)
        #     self.modelName = model_name
        #     self.model = TwoExpDecay()



