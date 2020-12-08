from core.liftetimeBenchmark.DecayGenerator import DecayGenerator
from core.analyze.lifetime import lifeTimeMeasurements, OneExpDecay, OneExpDecay_tail
import numpy as np
import matplotlib.pyplot as plt

class SingleBench:
    """
    Very simple and basic object to store the data of only one bench (the real paremeters of the decay and the fitted data).
    """
    def __init__(self, measurement, fitted_params_dict, aux_params_dict):
        # fit data and much more
        self.measurement = measurement
        # real parameters
        self.fitted_params_dict = fitted_params_dict
        # Non fitted auxiliaray parameters like noise, etc
        self.aux_params_dict = aux_params_dict
        # ?
        self.errors = {}

class LifeTimeBenchmark:
    def __init__(self):
        self.benchs = []
        self.results_dict = {}
        self.is_ini_guess = True
        self.is_fit = False

    def launch_benchmark(self):
        pass

    def create_data(self):
        # Generate the decay
        #FIXME UI and argument in the function
        nb_decay = 5
        decay_params_dict_ = {}
        # Min / max
        decay_params_dict_["tau"] = [1, 10]
        decay_params_dict_["noise"] = [0, 0]
        decay_params_dict_["nb_photon"] = [1000, 100000]
        decay_params_dict_["t0"] = [0.5, 0.5]
        decay_params_dict_["tau_irf"] = [0.5, 0.5]
        decay_params_dict_["shift_irf"] = [0, 0]
        decay_generator = DecayGenerator(nb_decay=nb_decay, time_nbins=4096, model="single_exp", params_dict=decay_params_dict_)
        decay_generator.generate_data()

        # Transfer the decay to "measurement" objects
        for decay in decay_generator.decays:
            measurement = lifeTimeMeasurements()
            measurement.data = decay.data
            measurement.time_axis = decay_generator.time_ns
            measurement.type = "lifetime"
            measurement.error_bar = np.sqrt(measurement.data + 1)
            #FIXME Model name should be in the method argument
            measurement.model = OneExpDecay(decay_generator.irf)
            measurement.model.modelName = "One Decay IRF"
            measurement.params = measurement.model.make_params()

            measurement.model.IRF = decay_generator.irf
            measurement.model.data = decay.data

            # MLE ou chi square
            measurement.qty_to_min = "auto"

            #FIXME -> in the argument
            measurement.fitting_method1 = "leastsq"
            measurement.fitting_method2 = "None"
            self.benchs.append(SingleBench(measurement, decay.fitted_params_dict, decay.aux_params_dict))

        # Fit all the bench
        for bench in self.benchs:
            if self.is_ini_guess:
                bench.measurement.model.guess(decay_generator.time_ns, bench.measurement.params)
            if self.is_fit:
                bench.measurement.fit()
            for param_key in bench.measurement.params:
                if param_key in bench.fitted_params_dict:
                    # FIXME how do you define error ?
                    bench.errors[param_key] = (bench.measurement.params[param_key] - bench.fitted_params_dict[param_key])**2

        # Gather the results
        for key in decay_generator.decays[0].fitted_params_dict:
            ini_vals, errors, ini_guess = [], [], []
            for bench in self.benchs:
                ini_vals.append(bench.fitted_params_dict[key])
                errors.append(bench.errors[key])
                ini_guess.append(bench.measurement.model.ini_params[key])

            self.results_dict[key] = [ini_vals, errors, ini_guess]


        print(self.results_dict)
        # Compress the data size

        # Exploit data

        # Give a score

        # Plots
        # Scatter or histogramm ?
        self.plot_results(output="graph.png")

    def plot_results(self, output=None):
        """Visualize the result of the brute force grid search.

        The output file will display the error value per parameter and contour
        plots for all combination of two parameters.

        Inspired by the `corner` package (https://github.com/dfm/corner.py).

        Parameters
        ----------
        result : :class:`~lmfit.minimizer.MinimizerResult`
            Contains the results from the :meth:`brute` method.

        output : str, optional
            Name of the output PDF file (default is 'None')
        """


        # self.top_level = tk.Toplevel(self.master_frame)
        # self.top_level.title("Explore chi2")

        npars_fitted = len(self.benchs[0].fitted_params_dict)
        npars_aux = len(self.benchs[0].aux_params_dict)
        #TODO dpi is function of the nb of graph
        self.fig, self.axes = plt.subplots(npars_fitted, npars_fitted + npars_aux, dpi=100)
        fig = self.fig
        axes = self.axes


        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.top_level)
        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.top_level)
        # self.canvas._tkcanvas.pack(side='top', fill='both', expand=1)

        varlabels_fitted = list(self.benchs[0].fitted_params_dict.keys())
        varlabels_aux = list(self.benchs[0].aux_params_dict.keys())

        for i, par1 in enumerate(varlabels_fitted):
            for j, par2 in enumerate(varlabels_fitted):

                # parameter vs error in case of only one parameter
                if npars_fitted == 1:
                    key = varlabels_fitted[i]
                    ax = axes[0]
                    ax.plot(self.results_dict[key][0], self.results_dict[key][1], 'o', ms=3)

                    ax.set_ylabel(r'Error')
                    ax.set_xlabel(key)

                # parameter vs errors profile on top
                elif i == j and j < npars_fitted - 1:
                    if i == 0:
                        axes[0, 0].axis('off')
                    ax = axes[i, j + 1]
                    # red_axis = tuple([a for a in range(npars_fitted) if a != i])
                    key = varlabels_fitted[i]
                    ax.plot(self.results_dict[key][0], self.results_dict[key][1], 'o', ms=3)
                    ax.set_ylabel(r'Error')
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.set_ticks_position('right')
                    ax.set_xticks([])

                # parameter vs errors profile on the left
                elif j == 0 and i > 0:
                    ax = axes[i, j]
                    # red_axis = tuple([a for a in range(npars_fitted) if a != i])

                    key = varlabels_fitted[i]
                    ax.plot(self.results_dict[key][0], self.results_dict[key][1], 'o', ms=3)
                    ax.invert_xaxis()
                    ax.set_ylabel(key)
                    if i != npars_fitted - 1:
                        ax.set_xticks([])
                    elif i == npars_fitted - 1:
                        ax.set_xlabel(r'Error')

                # contour plots for all combinations of two parameters
                elif j > i:
                    ax = axes[j, i + 1]
                    key_i = varlabels_fitted[i]
                    key_j = varlabels_fitted[j]

                    red_axis = tuple([a for a in range(npars_fitted) if a != i and a != j])
                    X, Y = np.meshgrid(self.results_dict[key_i][0],
                                       self.results_dict[key_j][0])
                    # lvls1 = np.linspace(result.brute_Jout.min(),
                    #                     np.median(result.brute_Jout) / 2.0, 7, dtype='int')
                    # lvls2 = np.linspace(np.median(result.brute_Jout) / 2.0,
                    #                     np.median(result.brute_Jout), 3, dtype='int')
                    # lvls = np.unique(np.concatenate((lvls1, lvls2)))
                    size = len(self.results_dict[key_i][0])
                    Z = np.zeros(size, size)

                    for i in range(size):
                        for j in range(size):
                            Z[i, j] = self.results_dict[key_i][1][i] + self.results_dict[key_j][1][j]

                    ax.contourf(X.T, Y.T, Z)
                    ax.set_yticks([])
                    if j != npars_fitted - 1:
                        ax.set_xticks([])
                    elif j == npars_fitted - 1:
                        ax.set_xlabel(key_i)
                    if j - i >= 2:
                        axes[i, j].axis('off')


        for i, par1 in enumerate(varlabels_aux):
            for j, par2 in enumerate(varlabels_fitted):
                pass
                # ax = axes[i, j]
                # key_i = varlabels_fitted[i]
                # key_j = varlabels_fitted[j]
                #
                # X, Y = np.meshgrid(self.results_dict[key_i][0],
                #                    self.results_dict[key_j][0])


        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

        if output is not None:
            self.fig.savefig(output)

        self.fig.show()


if __name__ == "__main__":

    life_time_benchmark = LifeTimeBenchmark()
    life_time_benchmark.create_data()

    print("OK")

