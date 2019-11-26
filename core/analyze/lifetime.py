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



class lifetimeModelClass():
    def __init__(self, IRF=None):
        self.IRF = IRF
        # self.x_range = None
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
        bckgnd_corrected_data = self.data - self.data_bckgnd
        bckgnd_corrected_data[bckgnd_corrected_data < 0] = 0
        self.observed_count = (bckgnd_corrected_data).sum()
        # self.observed_count = (self.data - self.data_bckgnd).sum()
        self.non_convoluted_decay = np.exp(-(t) / tau)
        # t_0 is in the shift

        if self.IRF is not None:
            IR = shift_scipy(self.IRF.processed_data, shift, mode='wrap')
            # IR = np.roll(self.IR, shift)
            conv = np.convolve(self.non_convoluted_decay, IR)[0:np.size(self.non_convoluted_decay)]
            # Test if sum is different from zero ?
            conv /= conv.sum()
            # return conv[self.x_range[0]:self.x_range[1]] + self.data_bckgnd
            return self.observed_count*conv + self.data_bckgnd
        else:
            self.non_convoluted_decay /= self.non_convoluted_decay.sum()
            return self.non_convoluted_decay + self.data_bckgnd

    def make_params(self):
        params = Parameters()
        params.add(name="tau", value=1, min=0.01, max=np.inf, brute_step=0.1)
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
        self.observed_count = (self.data - self.data_bckgnd).sum()
        self.non_convoluted_decay = a1*np.exp(-t/tau1) + (1-a1)*np.exp(-t/tau2)
        # t_0 is in the shift

        if self.IRF is not None:
            IR = shift_scipy(self.IRF.processed_data, shift, mode='wrap')
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

    def __init__(self, exps, exp, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, name="", comment=""):
        super().__init__(exps, exp, exp_param, num_channel, start_tick, end_tick, "lifetime", name, comment)
        self.IRF = None
        self.use_IR = False

    def set_additional_param_for_calculation(self, params):
        pass

    def calculate(self):
        timeStamps = self.exp.data.channels[self.num_channel].photons['timestamps']
        nanotimes = self.exp.data.channels[self.num_channel].photons['nanotimes']
        if self.start_tick == 0:
            start_tick = self.exp.data.channels[self.num_channel].start_tick
        else:
            start_tick = self.start_tick

        if self.end_tick == -1:
            end_tick = self.exp.data.channels[self.num_channel].end_tick
        else:
            end_tick = self.end_tick

        idxStart, idxEnd = np.searchsorted(timeStamps, (start_tick, end_tick))
        nanotimes = nanotimes[idxStart:idxEnd]

        self.create_histogramm(nanotimes)
        return self

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

    def set_model_IRF(self):
        self.model.IRF = self.IRF

    def set_use_IR(self, use_IR=False):
        self.use_IR = use_IR
        if self.model is not None:
            self.model.use_IR = use_IR

    def fit(self, params=None, mode="chi2"):
        # ATTENTION au -1 -> cela créé une case de moins dans le tableau.
        if self.model is not None:
            self.model.data = self.data

        if self.use_IR:
            self.set_params(params)

            # self.find_idx_of_fit_limit(idx_start, idx_end)
            y_eval_range  = self.data[self.idx_start:self.idx_end]
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
                # NB : we all the data on the time_axis in order to calculate the Convolution by the IRF.
                # But the likelihood has to be calculated only on the selection
                ymodel = self.model.eval(x, params)
                # likelyhood = 2 * (ydata*np.log(ydata/ymodel)).sum()
                likelyhood = -(2 * ydata[self.idx_start:self.idx_end] * np.log(ymodel[self.idx_start:self.idx_end])).sum()
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

            # TODO choose minimization technique
            minimization = "chi-square"
            # minimization = "maximum likelihood"
            nonzero_data = np.copy(self.data)
            # nonzero_data = nonzero_data[nonzero_data == 0] = 1
            nonzero_data[nonzero_data == 0] = 1
            weights = 1 / np.sqrt(nonzero_data)
            # weights[y == 0] = 1. / np.sqrt(baseline_true)
            weights[self.data == 0] = 0
            if minimization == "chi-square":
                self.fit_results = minimize(residuals, self.params, args=(self.time_axis, self.data, weights), method='nelder', iter_cb=callback_iteration, nan_policy='propagate')

                self.fit_results = minimize(residuals, self.fit_results.params, args=(self.time_axis, self.data, weights),
                                            method='leastsq', iter_cb=callback_iteration, nan_policy='propagate')
            elif minimization == "maximum likelihood":
                # self.fit_results = minimize(maximum_likelihood_method, self.params, args=(self.time_axis, self.data), method='nelder', iter_cb=callback_iteration)
                self.fit_results = minimize(maximum_likelihood_method, self.params, args=(self.time_axis, self.data), method='nelder', iter_cb=callback_iteration, nan_policy='propagate')

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

    def eval(self, params_=None):

        # Because of the convolution with the IRF, the eval function needs to access all the data and not only
        # the x-selected area

        self.set_params(params_)
        # y = self.data[self.idx_start:self.idx_end]
        # x_eval_range = self.time_axis[self.idx_start:self.idx_end]
        # self.model.x_range = (self.idx_start, self.idx_end)
        # self.model.observed_count = (self.data - self.params['bckgnd'].value).sum()
        self.model.data = self.data

        # x = self.time_axis[self.idx_start:self.idx_end]
        # y = self.data[self.idx_start:self.idx_end]

        # self.model.data = self.data
        self.eval_y_axis = self.model.eval(params=self.params, t=self.time_axis)

        self.eval_y_axis = self.eval_y_axis[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]
        y = self.data[self.idx_start:self.idx_end]

        self.residuals = self.eval_y_axis - y
        self.residual_x = x
        self.eval_x_axis = self.fit_x = x



    def explore_chi2_surf(self, params_):
        self.set_model(params_["model_name"])
        self.params = self.model.make_params()

        self.set_params(params_)
        self.model.data = self.data

        def fcn2min(params_, x, data, weights):
            ymodel = self.model.eval(params=params_, t=x)
            return (data[self.idx_start:self.idx_end] - ymodel[self.idx_start:self.idx_end]) * weights[self.idx_start:self.idx_end]

        self.model.observed_count = (self.data - self.params['bckgnd'].value).sum()

        nonzero_data = np.copy(self.data)
        nonzero_data[nonzero_data == 0] = 1
        weights = 1 / np.sqrt(nonzero_data)
        # weights[y == 0] = 1. / np.sqrt(baseline_true)
        weights[self.data == 0] = 0

        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]

        fitter = Minimizer(fcn2min, self.params, fcn_args=(self.time_axis, self.data, weights))
        result_brute = fitter.minimize(method='brute', Ns=25, keep=25)
        return result_brute

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

    def create_chronogram_overlay(self, chronogram, micro_1, micro_2, photon_threshold=5):
        if micro_1 > micro_2:
            micro_1, micro_2 = micro_2, micro_1
        micro_1 = int(micro_1 / (self.exp_param.mIcrotime_clickEquivalentIn_second*1E9))
        micro_2 = int(micro_2 / (self.exp_param.mIcrotime_clickEquivalentIn_second * 1E9))
        nanotimes = chronogram.get_raw_data("nanotimes")

        # Test if photon are in the interval
        nanotimes_mask = (nanotimes >= micro_1) & (nanotimes <= micro_2)

        nanotimes_mask = nanotimes_mask.astype(int)
        # Count for each chronogram bin the ratio photon inside the nanotimes boundaries vs total number of photon
        num_photon_edge = chronogram.get_num_photon_of_edge(is_absolute_number=False)

        multi_dim_mask = np.hsplit(nanotimes_mask, num_photon_edge)
        # multi_dim_selected_photon_by_bin = np.cumsum(multi_dim_mask, axis=1)
        selected_photon_by_bin = np.zeros(len(multi_dim_mask))
        for i, dim in enumerate(multi_dim_mask):
            selected_photon_by_bin[i] = np.sum(dim)

        # selected_photon_by_bin = np.ravel(multi_dim_selected_photon_by_bin)

        # selected_photon_by_bin = np.ravel(np.cumsum(np.hsplit(nanotimes_mask, num_photon_edge)))
        chrono = np.copy(chronogram.data)
        # can't divide by zero...
        chrono[chrono == 0] = 1
        overlay_data = selected_photon_by_bin / chrono
        overlay_data[chrono < photon_threshold] = 0
        return overlay_data

    # def set_params(self, params):
    #     if self.modelName == "One Decay":
    #         self.params['t0'].set(value=params[0],  vary=True, min=0, max=None)
    #         self.params['amp'].set(value=params[1], vary=True, min=0, max=None)
    #         self.params['tau'].set(value=params[2], vary=True, min=0, max=None)
    #         self.params['cst'].set(value=params[3], vary=True, min=0, max=None)
    #
    #     elif self.modelName == "One Decay IRF":
    #         self.params['tau'].set(value=params[0])
    #         self.params['shift'].set(value=params[1])
    #         self.params['bckgnd'].set(value=params[2])
    #         # self.model.bckgnd = params[2]
    #
    #     elif self.modelName == "One Decay Normalized":
    #         self.params['t0'].set(value=params[0],  vary=True, min=0, max=None)
    #         self.params['tau'].set(value=params[1], vary=True, min=0, max=None)
    #         self.params['cst'].set(value=params[2], vary=True, min=0, max=None)
    #
    #     elif self.modelName == "Two Decays IRF":
    #         self.params['tau1'].set(value=params[0])
    #         self.params['a1'].set(value=params[1])
    #         self.params['tau2'].set(value=params[2])
    #         self.params['shift'].set(value=params[3])
    #         self.params['bckgnd'].set(value=params[4])

    def set_model(self, model_name):
        if model_name == "One Decay IRF":
            self.modelName = model_name
            self.model = OneExpDecay(self.IRF)
            self.params = self.model.make_params()

        elif model_name == "One Decay Tail":
            self.modelName = model_name
            # self.model = OneExpDecay()
            # self.params = self.model.make_params(t0=0, amp=1, tau=1, cst=0)

        elif model_name == "Two Decays IRF":
            self.modelName = model_name
            self.model = TwoExpDecay(self.IRF)
            self.params = self.model.make_params()

        elif model_name == "Two Decays Tail":
            # print (modelName)
            self.modelName = model_name
            self.model = TwoExpDecay()

        self.model.use_IR = self.use_IR
        self.model.IR = self.IRF


class IRF:
    def __init__(self, exps, exp, file_path=None):
        self.exps = exps
        self.exp = exp
        self.exp_param = exp.exp_param
        self.raw_data, self.processed_data, self.name = None, None, None
        self.start, self.end = None, None
        self.bckgnd = 0
        self.shift = None
        self.time_axis, self.time_axis_processed = None, None
        if file_path is not None:
            self.get_data(file_path)

    def get_data(self, file_path):
        self.name, self.raw_data, self.time_axis = self.exps.get_IRF_from_file(file_path)

    def fit(self, ini_params):
        # TODO test
        self.model_IR = ExpConvGauss()
        self.params_fit_IR = self.model_IR.make_params(mu=ini_params[0], sig=ini_params[1], tau=ini_params[2])
        self.fit_results = self.model_IR.fit(self.processed_data, self.params_fit_IR, t=self.time_axis_processed)

        self.fit_y_axis = self.fit_results.best_fit
        self.fit_x_axis = self.time_axis_processed

        self.residuals = self.fit_results.residual
        return self.fit_results

    def process(self):
        """
        Shift in nb of microtime channel
        """
        idx_begin = int(self.start/100.0 * self.exp_param.nb_of_microtime_channel)
        idx_end = int(self.end/100.0 * self.exp_param.nb_of_microtime_channel)
        # shift = int(self.shift/100.0 * self.exp_param.nb_of_microtime_channel)
        # self.IR_processed = np.roll(self.IR_raw[idx_begin:idx_end], int(self.IR_shift/100.0*self.exp_param.nb_of_microtime_channel))
        if self.raw_data is not None:
            self.processed_data = self.raw_data[idx_begin:idx_end].astype(np.float64)
            # background removal
            self.processed_data -= self.bckgnd
            self.processed_data[self.processed_data < 0] = 0

            # We divide by the sum of the IR so that the convolution doesn't change the amplitude of the signal.
            self.processed_data /= float(self.processed_data.sum())
            self.time_axis_processed = self.time_axis[idx_begin:idx_end]
            return "OK"
        else:
            return "No IR was loaded"



