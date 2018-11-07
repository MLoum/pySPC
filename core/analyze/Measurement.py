import numpy as np
from lmfit import minimize, Parameters, Model
from lmfit.models import LinearModel, ExponentialModel


def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars


class Measurements:

    def __init__(self, data_=None, time_axis_= None):
        self.params = Parameters()
        self.modelName = ""
        self.model = None

        self.time_axis = time_axis_

        self.data = data_
        self.error_bar = None

        self.eval_x_axis = None
        self.eval_y_axis = None

        self.idx_start, self.idx_end = 0, -1

        self.residuals, self.fit_results = None, None

        self.canonic_fig, self.canonic_fig_ax = None, None

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
        if idx_end != -1:
            self.idx_end = np.searchsorted(self.time_axis, idx_end)

    def fit(self, idx_start=0, idx_end=-1):
        """

        :param idx_start:
        :param idx_end:
        :return:
        """
        self.find_idx_of_fit_limit(idx_start, idx_end)
        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]
        self.fit_results = self.model.fit(y, self.params, t=x)

        self.eval_y_axis = self.fit_results.best_fit
        self.eval_x_axis = x

        self.residuals = self.fit_results.residual

        #self.evalParams(idx_start, idx_end)
        return self.fit_results

    def evalParams(self, idx_start=0, idx_end=-1):
        """

        :param idx_start:
        :param idx_end:
        :return:
        """
        self.find_idx_of_fit_limit(idx_start, idx_end)

        x = self.time_axis[self.idx_start:self.idx_end]
        y = self.data[self.idx_start:self.idx_end]

        self.eval_y_axis = self.model.eval(self.params, t=x)
        self.residuals = self.eval_y_axis - y
        self.eval_x_axis = x

    def guess(self, idx_start=0, idx_end=-1):
        """
        Guess the parameters using the guess method of the lmfit Model class instance  (i.e. the member self.model)

        :param idx_start:
        :param idx_end:
        :return:
        """
        self.find_idx_of_fit_limit(idx_start, idx_end)

        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]

        self.params = self.model.guess(y, x)
        self.evalParams(idx_start, idx_end)

    def set_params(self, params):
        """
        "Virtual" Method that has to be explicited in child classes

        :param params:
        :return:
        """
        pass

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



