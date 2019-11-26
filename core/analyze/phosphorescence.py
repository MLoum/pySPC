import numpy as np
from lmfit import minimize, Parameters, Model, fit_report, Minimizer
from lmfit.models import LinearModel, ExponentialModel
import matplotlib.pyplot as plt


from scipy.special import erfc
from scipy.ndimage.interpolation import shift as shift_scipy

from .Measurement import Measurements
from .histogram import histogram

def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars



class PhosphoMeasurements(Measurements):

    def __init__(self, exps, exp, exp_param=None, num_channel=0,  start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exps, exp, exp_param, num_channel, start_tick, end_tick, "phosphorescence", name, comment)
        self.num_start = 0
        self.num_stop = 1
        self.time_step_micros = 10

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

        self.create_histogramm(timestamps_start, timestamps_stop, self.time_step_micros)
        return self

    def set_additional_param_for_calculation(self, params):
        self.num_start, self.num_stop, self.time_step_micros = params
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
        idx_start_exp = np.searchsorted(timestamps_stop, timestamps_start)
        single_exp_array = np.split(timestamps_stop, idx_start_exp)

        # For each single experiment, we reset the starting time at zero by removing the time offset timestamps_start[i]
        for i, exp in enumerate(single_exp_array):
            exp -= timestamps_start[i]

        # The macro time clock is around 25ns which is very very short compared to a phosophorescence decay (1ms - 10s)
        # Consequently we have to bin the data
        binning_factor = int(time_step_micros*1E-6/self.exp_param.mAcrotime_clickEquivalentIn_second)
        if binning_factor == 0:
            binning_factor = 1
        single_exp_array = [single_exp_array / binning_factor for x in single_exp_array ]

        single_exp_array_concatenated = np.concatenate(single_exp_array)

        # Then, we create the phosphorescence decay curve by adding all the sub experiment curve. We need to know
        # max_time_bin = 0
        # for exp in single_exp_array:
        #     if exp[-1] > max_time_bin:
        #         max_time_bin = exp[-1]

        # Finally, we compute
        self.data = np.bincount(single_exp_array_concatenated)
        self.time_axis = np.arange(0, self.data.size) * binning_factor * self.exp_param.mAcrotime_clickEquivalentIn_second * 1E-6   # in µs


    def create_canonic_graph(self, is_plot_error_bar=False, is_plot_text=False):
        self.canonic_fig, self.canonic_fig_ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0)
        ax = self.canonic_fig_ax
        ax[0].semilogy(self.time_axis, self.data, "ro", alpha=0.5)
        for a in ax:
            a.grid(True)
            a.grid(True, which='minor', lw=0.3)


        if self.fit_results is not None:
            ax[0].semilogy(self.time_axis, self.fit_results.best_fit, "b-", linewidth=3)
        if self.residuals is not None:
            ax[1].plot(self.time_axis, self.fit_results.residual, 'k')
            ym = np.abs(self.fit_results.residual).max()
            ax[1].set_ylim(-ym, ym)

        if is_plot_text:
            pass
            # TODO Changer le texte selon les modeles
            if self.modelName == "One Decay":
                pass
            elif self.modelName == "Two Decays":
                msg = ((r'$\tau_1$ = {tau1:.2f} ns' + '\n' + r'$\tau_2$ = {tau2:.0f} ns' + '\n' + r'$A_1$ = {A1:.0f}' + '\n' + r'$A_2$ = {A2:.0f}')
                       .format(tau1=self.fit_results.values['tau1'], tau2=self.fit_results.values['tau2'], A1=self.fit_results.values['amp1'],
                               A2=self.fit_results.values['amp2']))
            # ax[0].text(.75, .9, msg,
            #            va='top', ha='left', transform=ax[0].transAxes, fontsize=18)

        ax[0].set_ylabel('Occurence', fontsize=40)
        ax[1].set_ylabel('residuals', fontsize=20)
        ax[1].set_xlabel('time (ns)', fontsize=40)

        ax[0].tick_params(axis='both', which='major', labelsize=20)
        ax[0].tick_params(axis='both', which='minor', labelsize=8)

        ax[1].tick_params(axis='x', which='major', labelsize=20)
        ax[1].tick_params(axis='x', which='minor', labelsize=8)
        ax[1].tick_params(axis='x', which='major', labelsize=20)
        ax[1].tick_params(axis='x', which='minor', labelsize=8)

        return self.canonic_fig

    def set_model(self, model_name):

        if model_name == "One Decay Tail":
            self.modelName = model_name
            self.model = OneExpDecay()
            self.params = self.model.make_params(t0=0, amp=1, tau=1, cst=0)

        elif model_name == "Two Decays Tail":
            # print (modelName)
            self.modelName = model_name
            self.model = TwoExpDecay()

        self.model.use_IR = self.use_IR
        self.model.IR = self.IRF


