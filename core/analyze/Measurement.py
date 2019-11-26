import numpy as np
from lmfit import minimize, Parameters, Model, Minimizer
from lmfit.models import LinearModel, ExponentialModel


def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars


class Measurements:
    def __init__(self, exps, exp, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, type="", name="", comment=""):
        #FIXME exp_param from exp
        self.exp_param = exp_param
        self.exps = exps
        self.exp = exp

        self.type = type
        self.name = name
        self.comment = comment

        self.start_tick = start_tick
        self.end_tick = end_tick

        self.num_channel = num_channel
        self.nb_of_photon = 0
        
        #Fit
        self.params = Parameters()
        self.modelName = ""
        self.model = None
        self.eval_x_axis = None
        self.eval_y_axis = None
        self.residuals, self.fit_results = None, None
        self.idx_start, self.idx_end = 0, -1

        self.time_axis = None
        self.data = None
        self.error_bar = None

        self.is_error_bar_for_fit = True

        self.fit_x = None
        self.residual_x = None


        self.canonic_fig, self.canonic_fig_ax = None, None

    def calculate(self):
        """
        Performs the calculation of the measurement from the raw data
        :return:
        """
        pass

    def set_additional_param_for_calculation(self, params):
        pass

    def find_idx_of_fit_limit(self, idx_start, idx_end):
        """
        User selection of a a specific part of a graph is often based on a arbitraty value on the x axis
        This method find the correspoding index of the photon

        :param idx_start: lowest x value of the user selected interval. Default is 0
        :param idx_end: highest x value of the user selected interval. Default is -1, that is to say latest value
        :return:
        """
        if idx_start != 0:
            self.idx_start = np.searchsorted(self.time_axis, idx_start)
        else:
            self.idx_start = 0
        if idx_end != -1:
            self.idx_end = np.searchsorted(self.time_axis, idx_end)
        else:
            self.idx_end = len(self.time_axis)
        return self.idx_start, self.idx_end


    def fit(self, params=None, mode="chi2"):
        """

        :param idx_start:
        :param idx_end:
        :return:
        """
        if params is not None:
            self.set_params(params)

        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]

        if self.is_error_bar_for_fit:
            error_bar = self.error_bar[self.idx_start:self.idx_end]
            self.fit_results = self.model.fit(y, self.params, t=x, weights=error_bar)
        else:
            self.fit_results = self.model.fit(y, self.params, t=x)

        self.eval_y_axis = self.fit_results.best_fit
        self.eval_x_axis = self.fit_x = x

        self.residuals = self.fit_results.residual
        self.residual_x = x

        return self.fit_results.fit_report()

    def eval(self, params_=None):
        """

        :param idx_start:
        :param idx_end:
        :return:
        """
        if params_ is not None:
            self.set_params(params_)

        x = self.time_axis[self.idx_start:self.idx_end]
        y = self.data[self.idx_start:self.idx_end]

        # self.model.data = self.data
        self.eval_y_axis = self.model.eval(params=self.params, t=x)
        self.residuals = self.eval_y_axis - y
        self.residual_x = x
        self.eval_x_axis = self.fit_x = x

    def guess(self, params=None):
        """
        Guess the parameters using the guess method of the lmfit Model class instance  (i.e. the member self.model)

        :param idx_start:
        :param idx_end:
        :return:
        """
        if params is not None:
            self.set_params(params)

        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]

        self.params = self.model.guess(y, x)
        self.eval()

    def explore_chi2_surf(self, params_):

        self.set_params(params_)


        def fcn2min(params):
            x = self.time_axis[self.idx_start:self.idx_end]
            y = self.data[self.idx_start:self.idx_end]

            ymodel = self.model.eval(params=self.params, t=x)
            self.residuals = self.eval_y_axis - y

            return y[self.idx_start:self.idx_end] - ymodel[self.idx_start:self.idx_end]

        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]

        fitter = Minimizer(fcn2min, self.params)
        result_brute = fitter.minimize(method='brute', Ns=25, keep=25)
        return result_brute

    def set_params(self, params_):
        """
        Set parameters for fit
        :param params:
        :return:
        """
        x_start, x_end = params_["lim_fit"]
        self.find_idx_of_fit_limit(x_start, x_end)
        self.is_error_bar_for_fit = params_["use_error_bar"]

        for i, key in enumerate(self.params):
            self.params[key].set(value=params_["val"][i], min=params_["min"][i], max=params_["max"][i], vary=bool(params_["hold"][i]), brute_step=params_["brute_step"][i])


    # def set_hold_params(self, params_hold):
    #     for i, key in enumerate(self.params):
    #         self.params[key].set(vary=params_hold[i])
    #
    # def set_params_min(self, params_min):
    #     for i, key in enumerate(self.params):
    #         self.params[key].set(min=params_min[i])
    #
    # def set_params_max(self, params_max):
    #     for i, key in enumerate(self.params):
    #         self.params[key].set(max=params_max[i])


    def set_model(self, model_name):
        """
        "Virtual" Method that has to be explicited in child classes

        :param model_name:
        :return:
        """
        pass

    def export(self, file_path=None):
        """
        Export to an external text file
        "Virtual" Method that has to be explicited in child classes

        :return:
        """
        pass

    def get_info(self):
        """
        return a string with all the parameters of the measurement in a nicely formatted text
        :return:
        """
        return "TODO !"

    def log(self, msg, level="info"):
        if self.exps.logger is not None:
            if level == "info":
                self.exps.logger.info(msg)

    def get_raw_data(self, type="timestamp", mode="data"):

        timestamps = self.exp.data.channels[self.num_channel].photons['timestamps']
        nanotimes = self.exp.data.channels[self.num_channel].photons['nanotimes']

        idxStart, idxEnd = np.searchsorted(timestamps, (self.start_tick, self.end_tick))

        if type == "nanotimes":
            if mode == "data":
                return nanotimes[idxStart:idxEnd]
            elif mode == "full":
                return nanotimes

        if type == "timestamp":
            if mode == "data":
                return timestamps[idxStart:idxEnd]
            elif mode == "full":
                return timestamps


