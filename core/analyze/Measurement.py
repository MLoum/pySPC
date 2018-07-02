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

        self.timeAxis = time_axis_

        self.data = data_

        self.eval_x_axis = None
        self.eval_y_axis = None

        self.idxStart, self.idxEnd  = 0, -1

        self.residuals = None

        self.fitResults = None

    def find_idx_of_fit_limit(self, idx_start, idx_end):
        """
        User selection of a a specific part of a graph is often based on a arbitraty value on the x axis
        This method find the correspoding index of the photon

        :param idx_start: lowest x value of the user selected interval. Default is 0
        :param idx_end: highest x value of the user selected interval. Default is -1, that is to say latest value
        :return:
        """
        if idx_start != 0:
            self.idxStart = np.searchsorted(self.timeAxis, idx_start)
        if idx_end != -1:
            self.idxEnd = np.searchsorted(self.timeAxis, idx_end)

    def fit(self, idx_start=0, idx_end=-1):
        """

        :param idx_start:
        :param idx_end:
        :return:
        """
        self.find_idx_of_fit_limit(idx_start, idx_end)
        y = self.data[self.idxStart:self.idxEnd]
        x = self.timeAxis[self.idxStart:self.idxEnd]
        self.fitResults = self.model.fit(y, self.params, t=x)

        self.eval_y_axis = self.fitResults.best_fit
        self.eval_x_axis = x

        self.residuals = self.fitResults.residual

        #self.evalParams(idx_start, idx_end)
        return self.fitResults

    def evalParams(self, idx_start=0, idx_end=-1):
        """

        :param idx_start:
        :param idx_end:
        :return:
        """
        self.find_idx_of_fit_limit(idx_start, idx_end)

        x = self.timeAxis[self.idxStart:self.idxEnd]
        y = self.data[self.idxStart:self.idxEnd]

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

        y = self.data[self.idxStart:self.idxEnd]
        x = self.timeAxis[self.idxStart:self.idxEnd]

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



