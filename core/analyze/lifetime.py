import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel
import matplotlib.pyplot as plt


from .Measurement import Measurements
from .histogram import histogram

def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars

class OneExpDecay(Model):
    """A exponential decay with a shift in time, with four Parameters ``t0``, ``amp``, ``tau`` and ``cst``.

    Defined as:

    .. math::

        f(t; , t0, amp, tau, cst) = cst + amp * exp(-(t - t0) / tau)

    """

    def __init__(self, IR=None, use_IR=False, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):

        self.IR = IR
        self.use_IR = use_IR

        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def oneExpDecay(t, t0, amp, tau, cst):
            exp_decay = cst + amp * np.exp(-(t - t0) / tau)
            exp_decay[t < t0] = cst
            if self.use_IR:
                return np.convolve(exp_decay, self.IR)[0:np.size(exp_decay)]
            else:
                return exp_decay

        super(OneExpDecay, self).__init__(oneExpDecay, **kwargs)

    def guess(self, data, x=None, **kwargs):
        t0, amp, tau, cst = 0., 0., 0., 0.
        #if x is not None:
        idx_t0 = np.argmax(data)
        t0 = x[idx_t0]
        amp = np.max(data)
        #Searching for position where amp is divided by e=2.71
        subarray = data[idx_t0:]
        #tau = np.where(subarray < amp/np.exp(1))[0]
        idx_tau = np.argmax(subarray < (float) (amp) / np.exp(1))
        tau = x[idx_tau] - t0
        #TODO check if it is not the case
        cst = np.min(data[np.nonzero(data)]) #Attention aux canaux à zeros

        pars = self.make_params(t0=t0, amp=amp, tau=tau, cst=cst)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"


class TwoExpDecay(Model):
    """Two exponential decays with a shift in time, with four Parameters ``t0``, ``amp1``, ``tau1``, ''amp2'', ''tau5'' and ``cst``.

    Defined as:

    .. math::

        f(t; , t0, amp, tau, cst) = cst + amp1 * exp(-(t - t0) / tau1) + amp2 * exp(-(t - t0) / tau2)

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def twoExpDecay(t, t0, amp1, tau1, amp2, tau2, cst):
            if t < t0:
                return 0
            else:
                return cst + amp1 * np.exp(-(t - t0) / tau1) + amp2 * np.exp(-(t - t0) / tau2)

        super(TwoExpDecay, self).__init__(twoExpDecay, **kwargs)

    def guess(self, data, x=None, **kwargs):
        t0, amp1, tau1, amp2, tau2, cst = 0., 0., 0., 0., 0., 0.
        #if x is not None:
        idx_t0 = np.argmax(data)
        t0 = x[idx_t0]
        # TODO


        pars = self.make_params(t0=t0, amp=amp, tau=tau, cst=cst)
        return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"

#TODO creer une classe mère pour les analyses.
class lifeTimeMeasurements(Measurements):

    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "lifetime", name, comment)
        self.IR_raw, self.IR_processed = None, None
        self.IR_start, self.IR_end = None, None
        self.IR_shift = None
        self.IR_time_axis, self.IR_time_axis_processed = None, None
        self.use_IR = False

    def create_histogramm(self, nanotimes):
        self.data = np.zeros(self.exp_param.nb_of_microtime_channel, dtype=np.uint)
        # self.time_axis = np.arange(self.exp_param.nb_of_microtime_channel) * self.exp_param.mIcrotime_clickEquivalentIn_second*1E9
        self.data = np.bincount(nanotimes)
        self.time_axis = np.arange(0, self.data.size) * self.exp_param.mIcrotime_clickEquivalentIn_second*1E9
        self.trim_life_time_curve()

    def trim_life_time_curve(self):
        nonzero = np.nonzero(self.data)
        idxStartNonZero = nonzero[0][0]
        idxEndNonZero = nonzero[0][-1]
        self.data = self.data[idxStartNonZero:idxEndNonZero]
        self.time_axis = self.time_axis[idxStartNonZero:idxEndNonZero]

    def shift_histogramm(self, shift):
        self.data = np.roll(self.data, shift)

    def set_model_IR(self):
        self.model.IR = self.IR_processed

    def set_use_IR(self, use_IR=False):
        self.use_IR = use_IR
        if self.model is not None:
            self.model.use_IR = use_IR

    def process_IR(self):
        """
        Shift in nb of microtime channel
        """
        idx_begin = int( self.IR_start/100.0 *self.exp_param.nb_of_microtime_channel)
        idx_end = int(self.IR_end/100.0 * self.exp_param.nb_of_microtime_channel)
        shift = int(self.IR_shift/100.0*self.exp_param.nb_of_microtime_channel)
        # self.IR_processed = np.roll(self.IR_raw[idx_begin:idx_end], int(self.IR_shift/100.0*self.exp_param.nb_of_microtime_channel))
        if self.IR_raw is not None:
            self.IR_processed = self.IR_raw[idx_begin:idx_end]
            # We divide by the sum of the IR so that the convolution doesn't change the amplitude of the signal.
            self.IR_processed /= self.IR_processed.sum()
            if self.model is not None:
                self.model.IR = self.IR_processed
            self.IR_time_axis_processed = self.IR_time_axis[idx_begin - shift:idx_end - shift]
            return "OK"
        else:
            return "No IR was loaded"

    def generate_artificial_IR(self, mainWidth, secondaryWidth, secondaryAmplitude, timeOffset):
        self.IR = (1-secondaryWidth) * np.exp( - (self.eval_x_axis - timeOffset)**2/mainWidth) + secondaryAmplitude * np.exp( - (self.eval_x_axis - timeOffset)**2/secondaryWidth)
        #scale
        self.IR /= self.IR.sum()


    def create_canonic_graph(self, is_plot_error_bar=False, is_plot_text=False):
        self.canonic_fig, self.canonic_fig_ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0)
        ax = self.canonic_fig_ax
        ax[0].semilogy(self.time_axis, self.data, "ro", alpha=0.5)
        for a in ax:
            a.grid(True);
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

    def set_params(self, params):
        if self.modelName == "One Decay":
            self.params['t0'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['amp'].set(value=params[1], vary=True, min=0, max=None)
            self.params['tau'].set(value=params[2], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[3], vary=True, min=0, max=None)

        if self.modelName == "Two Decays":
            self.params['t0'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['amp1'].set(value=params[1], vary=True, min=0, max=None)
            self.params['tau1'].set(value=params[2], vary=True, min=0, max=None)
            self.params['amp2'].set(value=params[3], vary=True, min=0, max=None)
            self.params['tau2'].set(value=params[4], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[5], vary=True, min=0, max=None)


    def set_model(self, modelName):

        if modelName == "One Decay":
            self.modelName = modelName
            self.model = OneExpDecay()
            self.params = self.model.make_params(t0=0, amp=1, tau=1, cst=0)

        if modelName == "Two Decays":
            print (modelName)
            self.modelName = modelName
            self.model = TwoExpDecay()
            self.params = self.model.make_params(t0=0, amp1=1, tau1=1, amp2=1, tau2=1, cst=0)

        self.model.use_IR = self.use_IR
        self.model.IR = self.IR_processed





