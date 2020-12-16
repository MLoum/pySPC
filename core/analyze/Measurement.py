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
    def __init__(self, exps=None, exp=None, exp_param=None, num_channel=0, start_tick=0, end_tick=-1, type="", name="", comment=""):
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
        self.fit_results_method1, self.fit_results = None, None


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


    def fit(self, params=None):
        """
        :return:
        """
        if params is not None:
            self.set_params(params)

        y = self.data[self.idx_start:self.idx_end]
        x = self.time_axis[self.idx_start:self.idx_end]

        if self.is_error_bar_for_fit:
            error_bar = self.error_bar[self.idx_start:self.idx_end]
        else:
            error_bar = None
        self.fit_results_method1 = self.fit_results = self.model.fit(y, self.params, t=x, weights=error_bar, method=self.fitting_method1)


        if self.fitting_method2 != "None":
            self.fit_results = self.model.fit(y, self.fit_results.params, weights=error_bar, t=x, method=self.fitting_method2)

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
        self.fitting_method1 = params_["method1"]
        self.fitting_method2 = params_["method2"]
        self.qty_to_min = params_["qty_to_min"]

        for i, key in enumerate(self.params):
            # self.params[key].set(value=params_["val"][i], min=params_["min"][i], max=params_["max"][i], vary=bool(params_["hold"][i]), brute_step=params_["brute_step"][i])
            self.params[key].set(value=params_[key]["value"], min=params_[key]["min"], max=params_[key]["max"], vary=params_[key]["vary"], brute_step=params_[key]["b_step"])


    def set_model(self, model_name):
        """
        "Virtual" Method that has to be explicited in child classes

        :param model_name:
        :return:
        """
        pass

    def export(self, mode="text", file_path=None, x_selec_min=0, x_selec_max=-1):
        """
        Export to an external text file
        :return:
        """
        x = self.time_axis
        y = self.data
        data_fit = self.eval_y_axis
        data_residual = self.residuals
        if mode == "text":
            if data_fit is None:
                data = np.column_stack((x, y))
            else:
                #FIXME index ?
                x1 = np.searchsorted(x, x_selec_min)
                x2 = np.searchsorted(x, x_selec_max)

                x_selection_area = x[x1:x2]
                y_selection_area = y[x1:x2]
                data_fit_selection_area = data_fit[x1:x2]
                data_residual_selection_area = data_residual

                # export_size = min(x_selection_area.size, self.eval_y_axis.size)
                # data = np.column_stack((x_selection_area[0:export_size], y_selection_area[0:export_size], data_fit[0:export_size], data_residual[0:export_size]))
                data = np.column_stack((x_selection_area, y_selection_area,
                                        data_fit_selection_area, data_residual_selection_area))

            np.savetxt(file_path, data, header="x data fit residual")

        elif mode == "script":
            #TODO
            f = open(file_path.name, "w")
            header = "import matplotlib.pyplot as plt" \
                     "import numpy as np"
            f.writelines(header)

            f.writelines("self.figure = plt.Figure(figsize=figsize, dpi=dpi")
            f.writelines("self.ax = self.figure.add_axes([0.08, 0.3, 0.9, 0.65], xticklabels=[])")
            f.writelines("self.axResidual = self.figure.add_axes([0.08, 0.1, 0.9, 0.25])")

            # self.ax.tick_params(
            #     axis='x',  # changes apply to the x-axis
            #     which='both',  # both major and minor ticks are affected
            #     bottom=False,  # ticks along the bottom edge are off
            #     top=False,  # ticks along the top edge are off
            #     labelbottom=False)  # labels along the bottom edge are off"

            f.close()

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


