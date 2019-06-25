import numpy as np
from lmfit import minimize, Parameters, Model, fit_report
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



class lifetimeModelClass():
    def __init__(self, IRF=None):
        self.IRF = IRF
        self.x_range = None
        self.data = None
        self.non_convoluted_decay = None
        self.observed_count = 0

        # self.t0 = None
        self.data_bckgnd = None

    def guess(self):
        pass

    def eval(self, t, params):
        pass

class OneExpDecay(lifetimeModelClass):
    """A normalized exponential decay with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau, shift) = IRF(shift) x exp(-t / tau) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """
    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self):
        pass

    def eval(self, t,  params):
        tau = params['tau'].value
        shift = params['shift'].value
        self.data_bckgnd = params['bckgnd'].value
        self.non_convoluted_decay = np.exp(-(t) / tau)
        # t_0 is in the shift

        if self.IRF is not None:
            IR = shift_scipy(self.IRF, shift, mode='wrap')
            # IR = np.roll(self.IR, shift)
            conv = np.convolve(self.non_convoluted_decay, IR)[0:np.size(self.non_convoluted_decay)]
            conv /= conv.sum()
            # return conv[self.x_range[0]:self.x_range[1]] + self.data_bckgnd
            return self.observed_count*conv + self.data_bckgnd
        else:
            self.non_convoluted_decay /= self.non_convoluted_decay.sum()
            return self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="shift", value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params

class TwoExpDecay(lifetimeModelClass):
    """A normalized exponential decay with a shift in time, with two Parameters ``tau``, ''shift, and a fixed ``bckgnd``.

    Defined as:

    .. math::

        f(t; , tau1, a1, tau2, shift) = IRF(shift) x (a1 . exp(-t / tau1) + (1-a1) . exp(-t / tau2) ) + bckgnd

    The area under the decay curves obtained from the observed counts Cexp and from the predicted counts Ĉtheo must be
    conserved during optimization of the fitting parameters. Hence, the exponential does'nt have a amplitude parameter
    """
    def __init__(self, IRF=None):
        super().__init__(IRF)

    def guess(self):
        pass

    def eval(self, t,  params):
        tau1 = params['tau1'].value
        a1 = params['a1'].value
        tau2 = params['tau2'].value
        shift = params['shift'].value
        self.data_bckgnd = params['bckgnd'].value
        self.non_convoluted_decay = a1*np.exp(-(t) / tau1) + (1-a1)*np.exp(-(t) / tau2)
        # t_0 is in the shift

        if self.IRF is not None:
            IR = shift_scipy(self.IRF, shift, mode='wrap')
            # IR = np.roll(self.IR, shift)
            conv = np.convolve(self.non_convoluted_decay, IR)[0:np.size(self.non_convoluted_decay)]
            conv /= conv.sum()
            # return conv[self.x_range[0]:self.x_range[1]] + self.data_bckgnd
            return self.observed_count*conv + self.data_bckgnd
        else:
            self.non_convoluted_decay /= self.non_convoluted_decay.sum()
            return self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau1", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="a1", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="tau2", value=1, min=0, max=np.inf, brute_step=0.1)
        params.add(name="shift", value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        params.add(name="bckgnd", vary=False, value=0, min=-np.inf, max=np.inf, brute_step=0.1)
        return params

class ExpConvGauss(Model):
    """representing the convolution of a gaussian and an exponential. This function is used to model the IRF.

    Defined as:

    .. math::

        f(x;\mu,\sigma,\lambda) = \frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='propagate',
                 **kwargs):

        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def exgauss(t, mu, sig, tau):
            lam = 1. / tau
            return 0.5 * lam * np.exp(0.5 * lam * (2 * mu + lam * (sig ** 2) - 2 * t)) * \
                   erfc((mu + lam * (sig ** 2) - t) / (np.sqrt(2) * sig))

        super(ExpConvGauss, self).__init__(exgauss, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pass
        # return update_param_vals(pars, self.prefix, **kwargs)

    # __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
    __init__.__doc__ = "TODO"
    guess.__doc__ = "TODO"



#TODO creer une classe mère pour les analyses.
class lifeTimeMeasurements(Measurements):

    def __init__(self, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exp_param, num_channel, start_tick, end_tick, "lifetime", name, comment)
        self.IR_raw, self.IR_processed, self.IR_name = None, None, None
        self.IR_start, self.IR_end = None, None
        self.IR_bckg = 0
        self.IR_shift = None
        self.IR_time_axis, self.IR_time_axis_processed = None, None
        self.use_IR = False

    def create_histogramm(self, nanotimes):
        # self.data = np.zeros(self.exp_param.nb_of_microtime_channel, dtype=np.uint)
        # self.time_axis = np.arange(self.exp_param.nb_of_microtime_channel) * self.exp_param.mIcrotime_clickEquivalentIn_second*1E9
        self.data = np.bincount(nanotimes)
        self.time_axis = np.arange(0, self.data.size) * self.exp_param.mIcrotime_clickEquivalentIn_second*1E9
        self.trim_life_time_curve()
        if self.model is not None:
            self.model.total_count = self.data.sum()


    def trim_life_time_curve(self):
        nonzero = np.nonzero(self.data)
        idxStartNonZero = nonzero[0][0]
        idxEndNonZero = nonzero[0][-1]
        self.data = self.data[idxStartNonZero:idxEndNonZero]
        self.time_axis = self.time_axis[idxStartNonZero:idxEndNonZero]

    def shift_histogramm(self, shift):
        self.data = np.roll(self.data, shift)

    def set_IR(self, name, data, time_axis):
        """
        Fill the raw data of the IR, typically from a special IR measurement spc file.
        :param name: name of the file
        :param data: microtime Histogramm of the IR (y axis)
        :param time_axis: microtime axis in ns.
        :return:
        """
        self.IR_name = name
        self.IR_raw = np.copy(data)
        self.IR_time_axis = np.copy(time_axis)

    def set_model_IRF(self):
        self.model.IRF = self.IR_processed

    def set_use_IR(self, use_IR=False):
        self.use_IR = use_IR
        if self.model is not None:
            self.model.use_IR = use_IR

    def process_IR(self):
        """
        Shift in nb of microtime channel
        """
        idx_begin = int(self.IR_start/100.0 * self.exp_param.nb_of_microtime_channel)
        idx_end = int(self.IR_end/100.0 * self.exp_param.nb_of_microtime_channel)
        shift = int(self.IR_shift/100.0 * self.exp_param.nb_of_microtime_channel)
        # self.IR_processed = np.roll(self.IR_raw[idx_begin:idx_end], int(self.IR_shift/100.0*self.exp_param.nb_of_microtime_channel))
        if self.IR_raw is not None:
            self.IR_processed = self.IR_raw[idx_begin:idx_end].astype(np.float64)
            # background removal
            self.IR_processed -= self.IR_bckg
            self.IR_processed[self.IR_processed < 0] = 0

            # We divide by the sum of the IR so that the convolution doesn't change the amplitude of the signal.
            self.IR_processed /= float(self.IR_processed.sum())
            if self.model is not None:
                self.model.IRF = self.IR_processed
            self.IR_time_axis_processed = self.IR_time_axis[idx_begin - shift:idx_end - shift]
            return "OK"
        else:
            return "No IR was loaded"

    def generate_artificial_IR(self, mainWidth, secondaryWidth, secondaryAmplitude, timeOffset):
        # FIXME
        self.IR = (1-secondaryWidth) * np.exp( - (self.eval_x_axis - timeOffset)**2/mainWidth) + secondaryAmplitude * np.exp( - (self.eval_x_axis - timeOffset)**2/secondaryWidth)
        #scale
        self.IR /= self.IR.sum()

    def fit_IR(self, ini_params):
        self.model_IR = ExpConvGauss()
        self.params_fit_IR = self.model_IR.make_params(mu=ini_params[0], sig=ini_params[1], tau=ini_params[2])
        self.fit_IR_results = self.model_IR.fit(self.IR_processed, self.params_fit_IR, t=self.IR_time_axis_processed)

        self.fit_IR_y_axis = self.fit_results.best_fit
        self.fit_IR_x_axis = self.IR_time_axis_processed

        self.residuals = self.fit_results.residual
        return self.fit_IR_results

    def fit(self, idx_start=0, idx_end=-1):
        # ATTENTION au -1 -> cela créé une case de moins dans le tableau.
        if self.model is not None:
            self.model.data = self.data

        if self.use_IR:
            self.find_idx_of_fit_limit(idx_start, idx_end)
            y = self.data[self.idx_start:self.idx_end]
            x_eval_range = self.time_axis[self.idx_start:self.idx_end]
            self.model.x_range = (self.idx_start, self.idx_end)

            self.model.observed_count = (self.data - self.params['bckgnd'].value).sum()
            #TODO minimize MLC

            def residuals(params, x, y, weights):
                """
                Returns the array of residuals for the current parameters.
                """
                # tau = params['tau'].value
                # baseline = params['baseline'].value
                # offset = params['offset'].value
                # ampl = params['ampl'].value
                ymodel = self.model.eval(x, params)
                return (y[self.idx_start:self.idx_end] - ymodel[self.idx_start:self.idx_end]) * weights[self.idx_start:self.idx_end]

            def loglike(params, x, ydata):
                # tau, baseline, offset, ampl = params
                ymodel = self.model.eval(x, params)
                return (ymodel - ydata * np.log(ymodel)).sum()

            def maximum_likelihood_method(params, x, ydata):
                ymodel = self.model.eval(x, params)
                # likelyhood = 2 * (ydata*np.log(ydata/ymodel)).sum()
                likelyhood = -(2 * ydata * np.log(ymodel)).sum()
                return likelyhood

            def callback_iteration(params, iter, resid, *args, **kws):
                """
                Function to be called at each fit iteration. This function should have the signature
                iter_cb(params, iter, resid, *args, **kws), where params will have the current parameter values,
                iter the iteration number, resid the current residual array, and *args and **kws as passed
                to the objective function.
                :param params:
                :param iter:
                :param resid:
                :param args:
                :param kws:
                :return:
                """
                # TODO draw and report progress
                pass

            minimization = "chi-square"
            minimization = "maximum likelihood"
            weights = 1 / np.sqrt(self.data)
            # weights[y == 0] = 1. / np.sqrt(baseline_true)
            weights[self.data == 0] = 0
            if minimization == "chi-square":
                self.fit_results = minimize(residuals, self.params, args=(self.time_axis, self.data, weights), method='nelder', iter_cb=callback_iteration)

                self.fit_results = minimize(residuals, self.fit_results.params, args=(self.time_axis, self.data, weights),
                                            method='leastsq', iter_cb=callback_iteration)
            elif minimization == "maximum likelihood":
                self.fit_results = minimize(maximum_likelihood_method, self.params, args=(self.time_axis, self.data), method='nelder', iter_cb=callback_iteration)

                # self.fit_results = minimize(maximum_likelihood_method, self.fit_results.params, args=(self.time_axis, self.data),
                #                             method='leastsq', iter_cb=callback_iteration)


            # Neald


            #first approach with Nelder
            # self.fit_results = self.model.fit(y, self.params, t=self.time_axis, method='Nelder', weights=weights, iter_cb=None)
            #
            # self.fit_results = self.model.fit(y, self.params, t=self.time_axis, method='leastsq', weights=weights,
            #                                   iter_cb=None)

            #FIXME do eval_y_axis and eval_x_axis still in use ?
            self.eval_y_axis = self.model.eval(self.time_axis, self.fit_results.params)
            self.eval_x_axis = x_eval_range

            self.residuals = self.data[self.idx_start:self.idx_end] - self.eval_y_axis[self.idx_start:self.idx_end]
            self.fit_x = self.time_axis
            self.residual_x = self.time_axis[self.idx_start:self.idx_end]



            return fit_report(self.fit_results)
        else:
            return super().fit(idx_start, idx_end)

    def guess(self, idx_start=0, idx_end=-1):
        if self.model is not None:
            self.model.data = self.data

        if self.use_IR:
            self.find_idx_of_fit_limit(idx_start, idx_end)

            # y = self.data[self.idx_start:self.idx_end]
            # x_eval_range = self.time_axis[self.idx_start:self.idx_end]

            self.params = self.model.guess(self.data, self.time_axis)
            self.eval(idx_start, idx_end)
        else:
            super().guess(idx_start, idx_end)

    def eval(self, idx_start=0, idx_end=-1):
        if self.model is not None:
            self.model.data = self.data

        if self.use_IR:
            self.find_idx_of_fit_limit(idx_start, idx_end)
            y = self.data[self.idx_start:self.idx_end]
            x_eval_range = self.time_axis[self.idx_start:self.idx_end]
            self.model.x_range = (self.idx_start, self.idx_end)

            self.model.observed_count = (self.data - self.params['bckgnd'].value).sum()

            # self.eval_y_axis = self.model.eval(self.params, t=self.time_axis)
            # self.residuals = self.eval_y_axis - y

            ymodel = self.model.eval(self.time_axis, self.params)

            # self.find_idx_of_fit_limit(idx_start, idx_end)
            #
            # x_eval_range = self.time_axis[self.idx_start:self.idx_end]
            # y = self.data[self.idx_start:self.idx_end]
            #
            # self.model.x_range = (self.idx_start, self.idx_end)
            #
            # self.eval_y_axis = self.model.eval(self.params, t=self.time_axis)
            self.eval_y_axis = ymodel
            self.residuals = self.eval_y_axis[self.idx_start:self.idx_end] - self.data[self.idx_start:self.idx_end]
            self.eval_x_axis = self.time_axis

            self.fit_x = self.time_axis
            self.residual_x = self.time_axis[self.idx_start:self.idx_end]

        else:
            super().eval(idx_start, idx_end)

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

        elif self.modelName == "One Decay IRF":
            self.params['tau'].set(value=params[0])
            self.params['shift'].set(value=params[1])
            self.params['bckgnd'].set(value=params[2])
            # self.model.bckgnd = params[2]

        elif self.modelName == "One Decay Normalized":
            self.params['t0'].set(value=params[0],  vary=True, min=0, max=None)
            self.params['tau'].set(value=params[1], vary=True, min=0, max=None)
            self.params['cst'].set(value=params[2], vary=True, min=0, max=None)

        elif self.modelName == "Two Decays IRF":
            self.params['tau1'].set(value=params[0])
            self.params['a1'].set(value=params[1])
            self.params['tau2'].set(value=params[2])
            self.params['shift'].set(value=params[3])
            self.params['bckgnd'].set(value=params[4])

    def set_model(self, model_name):
        if model_name == "One Decay IRF":
            self.modelName = model_name
            self.model = OneExpDecay(self.IR_processed)
            self.params = self.model.make_params()

        elif model_name == "One Decay Tail":
            self.modelName = model_name
            # self.model = OneExpDecay()
            # self.params = self.model.make_params(t0=0, amp=1, tau=1, cst=0)

        elif model_name == "Two Decays IRF":
            self.modelName = model_name
            self.model = TwoExpDecay(self.IR_processed)
            self.params = self.model.make_params()

        elif model_name == "Two Decays Tail":
            # print (modelName)
            self.modelName = model_name
            self.model = TwoExpDecay()


        self.model.use_IR = self.use_IR
        self.model.IR = self.IR_processed





