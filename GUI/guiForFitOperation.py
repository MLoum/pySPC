import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from matplotlib.colors import LogNorm

class guiForFitOperation():


    def __init__(self, master_frame, controller, model_names, nb_param_fit=8, fitModeName="", is_burst_analysis=False):
        self.master_frame = master_frame
        self.model_names = model_names
        self.controller = controller
        self.view = self.controller.view

        self.measurement = controller.current_measurement

        self.list_label_param_fit = []

        self.is_burst_analysis = is_burst_analysis

        self.fit_mode_name = fitModeName

        self.nb_max_param_fit = nb_param_fit

    def populate(self):
        # cmd
        self.cmd_frame = ttk.Frame(self.master_frame)

        label = ttk.Label(self.cmd_frame, text='Model')
        label.grid(row=0, column=0)

        self.cb_model_sv = tk.StringVar()
        cb = ttk.Combobox(self.cmd_frame, width=15, justify=tk.CENTER, textvariable=self.cb_model_sv,
                          values='', state='readonly')
        cb.bind('<<ComboboxSelected>>', self.change_model)
        cb['values'] = self.model_names
        self.cb_model_sv.set(self.model_names[0])
        cb.set(self.model_names[0])
        cb.grid(row=0, column=1)

        ttk.Button(self.cmd_frame, text="IniGuess", command=self.ini_guess_fit).grid(row=1, column=0)
        ttk.Button(self.cmd_frame, text="Eval", command=self.eval_fit).grid(row=1, column=1)
        ttk.Button(self.cmd_frame, text="Fit", command=self.fit).grid(row=1, column=2)

        # Formula
        self.formulaFrame = tk.Frame(master=self.cmd_frame)
        self.formulaFrame.grid(row=3, column=0, columnspan=3)

        self.figTex = plt.Figure(figsize=(13, 2), dpi=28, frameon=False)
        self.axTex = self.figTex.add_axes([0, 0, 1, 1])

        self.axTex.axis('off')


        self.canvasTk = FigureCanvasTkAgg(self.figTex, master=self.formulaFrame)
        self.canvasTk.get_tk_widget().pack(side='top', fill='both', expand=1)

        # fit boundaries
        ttk.Label(self.cmd_frame, text='fit boundaries (x1, x2)').grid(row=4, column=0)


        self.idx_lim_for_fit_min_sv = tk.StringVar()
        ttk.Entry(self.cmd_frame, textvariable=self.idx_lim_for_fit_min_sv, justify=tk.CENTER, width=12).grid(row=4, column=1)

        self.idx_lim_for_fit_max_sv = tk.StringVar()
        ttk.Entry(self.cmd_frame, textvariable=self.idx_lim_for_fit_max_sv, justify=tk.CENTER, width=12).grid(row=4, column=2)

        # Quantity to minimize
        ttk.Label(self.cmd_frame, text='Qty to minimize').grid(row=5, column=0)
        self.cb_minqty_to_min_sv = tk.StringVar()
        cb = ttk.Combobox(self.cmd_frame, width=15, justify=tk.CENTER, textvariable=self.cb_minqty_to_min_sv,
                          values='', state='readonly')
        cb.bind('<<ComboboxSelected>>', self.change_minqty_to_min)
        cb['values'] = ["auto", "chi2", "max. likelyhood (MLE)"]
        self.cb_minqty_to_min_sv.set("auto")
        cb.set("auto")
        cb.grid(row=5, column=1)

        # Methods
        ttk.Label(self.cmd_frame, text='Method 1').grid(row=6, column=0)
        self.cb_method1_sv = tk.StringVar()
        cb = ttk.Combobox(self.cmd_frame, width=15, justify=tk.CENTER, textvariable=self.cb_method1_sv,
                          values='', state='readonly')
        cb.bind('<<ComboboxSelected>>', self.change_method1)
        cb['values'] = ["least_squares", "leastsq", "differential_evolution", "brute", "basinhopping", "ampgo", "nelder", "lbfgsb", "powell", "cg", "newton", "cobyla", "bfgs", "tnc", "trust-ncg", "trust-exact", "trust-krylov", "trust-constr", "dogleg", "slsqp", "emcee", "shgo", "dual_annealing"]
        self.cb_method1_sv.set("least_squares")
        cb.set("least_squares")
        cb.grid(row=6, column=1)

        ttk.Label(self.cmd_frame, text='Method 2').grid(row=6, column=2)
        self.cb_method2_sv = tk.StringVar()
        cb = ttk.Combobox(self.cmd_frame, width=15, justify=tk.CENTER, textvariable=self.cb_method2_sv,
                          values='', state='readonly')
        cb.bind('<<ComboboxSelected>>', self.change_method2)
        cb['values'] = ["None", "leastsq", "least_squares", "differential_evolution", "brute", "basinhopping", "ampgo", "nelder", "lbfgsb", "powell", "cg", "newton", "cobyla", "bfgs", "tnc", "trust-ncg", "trust-exact", "trust-krylov", "trust-constr", "dogleg", "slsqp", "emcee", "shgo", "dual_annealing"]
        self.cb_method2_sv.set("None")
        cb.set("None")
        cb.grid(row=6, column=3)

        # Explore chi square
        ttk.Button(self.cmd_frame, text="Explore χ²", command=self.explore_chi_square).grid(row=7, column=0)

        self.is_take_account_error_bar = tk.IntVar(value=1)
        ttk.Checkbutton(self.cmd_frame, text="Use error bars", variable=self.is_take_account_error_bar, command=self.use_error_bar).grid(row=8, column=0)

        self.cmd_frame.pack(side="left", fill="both", expand=True)

        # Parameters
        self.param_frame = ttk.Frame(self.master_frame)

        # column header
        ttk.Label(self.param_frame, text='').grid(row=0, column=0)
        ttk.Label(self.param_frame, text='value').grid(row=0, column=1)
        ttk.Label(self.param_frame, text='').grid(row=0, column=2)
        ttk.Label(self.param_frame, text='').grid(row=0, column=3)
        ttk.Label(self.param_frame, text='min').grid(row=0, column=4)
        ttk.Label(self.param_frame, text='max').grid(row=0, column=5)
        ttk.Label(self.param_frame, text='b step').grid(row=0, column=6)
        ttk.Label(self.param_frame, text='vary').grid(row=0, column=7)

        # Labels for parameters
        self.list_label_sv_param = [tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()]
        for i in range(self.nb_max_param_fit):
            ttk.Label(self.param_frame, text="", textvariable=self.list_label_sv_param[i]).grid(row=1+i, column=0)

        self.entry_text_size = 10


        # model_name = self.cb_model_sv.get()
        # self.measurement.set_model(model_name)
        # self.create_gui_from_measurement_params()
        self.change_model(None)

        ttk.Button(self.param_frame, text="Copy from fit", width=15, command=self.copy_param_from_fit).grid(row=self.nb_max_param_fit + 1, column=0, columnspan=7)

        self.param_frame.pack(side="left", fill="both", expand=True)

    def enable_disable_ui(self, num_end):
        for i in range(num_end):
            self.list_entry_param_fit[i].state(['!disabled'])
            self.list_entry_param_fit_min[i].state(['!disabled'])
            self.list_entry_param_fit_max[i].state(['!disabled'])
            self.list_button_plus[i].config(state="normal")
            self.list_button_minus[i].config(state="normal")
            self.list_entry_brute_step[i].config(state="normal")
            self.list_param_is_fixed[i].config(state="normal")

        for i in range(num_end, self.nb_max_param_fit):
            self.list_label_sv_param[i].set("")
            self.list_entry_param_fit[i].state(['disabled'])
            self.list_entry_param_fit_min[i].state(['disabled'])
            self.list_entry_param_fit_max[i].state(['disabled'])
            self.list_button_plus[i].config(state=tk.DISABLED)
            self.list_button_minus[i].config(state=tk.DISABLED)
            self.list_entry_brute_step[i].state(['disabled'])
            self.list_param_is_fixed[i].config(state=tk.DISABLED)

    def set_fit_formula(self, formula, fontsize=40):
        formula = "$" + formula + "$"

        self.axTex.clear()
        self.axTex.text(0, 0.2, formula, fontsize=fontsize)
        self.canvasTk.draw()

    def create_gui_from_measurement_params(self):
        self.gui_param_dict = {}
        self.gui_param_widget_dict = {}
        # self.changeModel(None)
        i = 0
        for key in self.measurement.params.keys():
            # Set label for param name
            self.list_label_sv_param[i].set(key)

            # Set current characteritics of the parameter
            param = self.measurement.params[key]
            # value, min, max, b step, hold
            self.gui_param_dict[key] = {}
            self.gui_param_dict[key]["value"] = tk.StringVar(value=str(param.value))
            self.gui_param_dict[key]["min"] = tk.StringVar(value=str(param.min))
            self.gui_param_dict[key]["max"] = tk.StringVar(value=str(param.max))
            self.gui_param_dict[key]["b_step"] = tk.StringVar(value=str(param.brute_step))
            self.gui_param_dict[key]["vary"] = tk.IntVar(value=int(param.vary))

            # Create corresponding widget (entry/button/checkbox)
            self.gui_param_widget_dict[key] = {}
            e = ttk.Entry(self.param_frame, textvariable=self.gui_param_dict[key]["value"], justify=tk.CENTER,
                      width=self.entry_text_size)
            e.grid(row=1 + i, column=1)
            self.gui_param_widget_dict[key]["e_value"] = e

            e = ttk.Entry(self.param_frame, textvariable=self.gui_param_dict[key]["min"], justify=tk.CENTER,
                      width=self.entry_text_size)
            e.grid(row=1 + i, column=4)
            self.gui_param_widget_dict[key]["e_min"] = e

            e = ttk.Entry(self.param_frame, textvariable=self.gui_param_dict[key]["max"], justify=tk.CENTER,
                      width=self.entry_text_size)
            e.grid(row=1 + i, column=5)
            self.gui_param_widget_dict[key]["e_max"] = e

            e = ttk.Entry(self.param_frame, textvariable=self.gui_param_dict[key]["b_step"], justify=tk.CENTER,
                      width=self.entry_text_size)
            e.grid(row=1 + i, column=6)
            self.gui_param_widget_dict[key]["e_b_step"] = e

            # hold check button
            cb = ttk.Checkbutton(self.param_frame, variable=self.gui_param_dict[key]["vary"])
            cb.grid(row=1 + i, column=7)
            self.gui_param_widget_dict[key]["cb_vary"] = cb

            # Set button + and -
            b = tk.Button(master=self.param_frame, text='+', command=lambda: self.value_plus(self.gui_param_dict[key]["value"]))
            b.grid(row=1 + i, column=2)
            self.gui_param_widget_dict[key]["b_plus"] = b
            b = tk.Button(master=self.param_frame, text='-', command=lambda: self.value_minus(self.gui_param_dict[key]["value"]))
            b.grid(row=1 + i, column=3)
            self.gui_param_widget_dict[key]["b_minus"] = b

            i += 1

    def change_model(self, event):
        model_name = self.cb_model_sv.get()

        self.measurement.set_model(model_name)
        self.create_gui_from_measurement_params()
        self.set_fit_formula(self.measurement.model.fit_formula)

    def copy_param_from_fit(self, params=None):
        if params is None:
            params = self.measurement.params

        for key in params.keys():
            param = params[key]
            self.gui_param_dict[key]["value"].set(param.value)



    def change_method1(self, event):
        pass

    def change_method2(self, event):
        pass

    def change_minqty_to_min(self, event):
        pass

    def explore_chi_square(self):
        params = self.get_fit_params()
        result = self.measurement.explore_chi2_surf(params)
        self.plot_results_brute(result)

    def set_bckgnd_cursor(self):
        pass

    def use_error_bar(self):
        is_use = self.is_take_account_error_bar.get()
        self.controller.set_use_error_bar(is_use)
        return is_use

    def ask_controller_eval_guess_fit(self, mode):
        params = self.get_fit_params()
        self.controller.guess_eval_fit(mode, params,
                                       is_burst_analysis=self.is_burst_analysis)

    def eval_fit(self):
        self.ask_controller_eval_guess_fit("eval")

    def ini_guess_fit(self):
        self.ask_controller_eval_guess_fit("guess")

    def fit(self):
        self.ask_controller_eval_guess_fit("fit")

    def value_plus(self, tk_string_var):
        # FIXME
        #TODO try execpt
        value = float(tk_string_var.get())
        value *= 1.1
        tk_string_var.set(str(value))

    def value_minus(self, tk_string_var):
        #FIXME
        tk_string_var.set(str(float(tk_string_var.get())*0.9))
        value = float(tk_string_var.get())
        value /= 1.1
        tk_string_var.set(str(value))


    def get_lim_for_fit(self):
        if self.idx_lim_for_fit_min_sv.get() == "":
            xlim_min_fit = 0
        else:
            xlim_min_fit = float(self.idx_lim_for_fit_min_sv.get())

        if self.idx_lim_for_fit_max_sv.get() == "":
            xlim_max_fit = -1
        else:
            xlim_max_fit = float(self.idx_lim_for_fit_max_sv.get())
        return (xlim_min_fit, xlim_max_fit)

    # def get_params_values(self):
    #     params_value = []
    #     for sv in self.list_entry_string_variable_fit:
    #         strValue = sv.get()
    #         if strValue == "":
    #             params_value.append(0)
    #         else:
    #             params_value.append(float(sv.get()))
    #     return params_value
    #
    # def get_params_min(self):
    #     params_min = []
    #     for sv in self.list_entry_string_variable_fit_min:
    #         strValue = sv.get()
    #         if strValue == "":
    #             # FIXME np.inf cause problems
    #             # params_min.append(-np.inf)
    #             params_min.append(-1E12)
    #         else:
    #             params_min.append(float(sv.get()))
    #     return params_min
    #
    # def get_params_max(self):
    #     params_max = []
    #     for sv in self.list_entry_string_variable_fit_max:
    #         strValue = sv.get()
    #         if strValue == "":
    #             # FIXME np.inf cause problems
    #             # params_max.append(np.inf)
    #             params_max.append(1E12)
    #         else:
    #             params_max.append(float(sv.get()))
    #     return params_max



    def get_fit_params(self):
        params = {}
        params["model_name"] = self.cb_model_sv.get()
        params["method1"] = self.cb_method1_sv.get()
        params["method2"] = self.cb_method2_sv.get()
        params["qty_to_min"] = self.cb_minqty_to_min_sv.get()

        for key in self.measurement.params.keys():
            params[key] = {}
            params[key]["value"] = float(self.gui_param_dict[key]["value"].get())
            params[key]["min"] = float(self.gui_param_dict[key]["min"].get())
            params[key]["max"] = float(self.gui_param_dict[key]["max"].get())
            params[key]["b_step"] = float(self.gui_param_dict[key]["b_step"].get())
            params[key]["vary"] = bool(self.gui_param_dict[key]["vary"].get())

        # params["val"] = self.get_params_values()
        # params["min"] = self.get_params_min()
        # params["max"] = self.get_params_max()
        # params["hold"] = self.get_params_hold()
        params["lim_fit"] = self.get_lim_for_fit()
        # params["brute_step"] = self.get_brute_step()
        params["use_error_bar"] = self.use_error_bar()
        return params

    def plot_results_brute(self, result, best_vals=True, varlabels=None,
                           output=None):
        """Visualize the result of the brute force grid search.

        The output file will display the chi-square value per parameter and contour
        plots for all combination of two parameters.

        Inspired by the `corner` package (https://github.com/dfm/corner.py).

        Parameters
        ----------
        result : :class:`~lmfit.minimizer.MinimizerResult`
            Contains the results from the :meth:`brute` method.

        best_vals : bool, optional
            Whether to show the best values from the grid search (default is True).

        varlabels : list, optional
            If None (default), use `result.var_names` as axis labels, otherwise
            use the names specified in `varlabels`.

        output : str, optional
            Name of the output PDF file (default is 'None')
        """


        self.top_level = tk.Toplevel(self.master_frame)
        self.top_level.title("Explore chi2")

        npars = len(result.var_names)
        #TODO dpi is funciton of the nb of graph
        self.fig, self.axes = plt.subplots(npars, npars, dpi=100)
        fig = self.fig
        axes = self.axes



        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top_level)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top_level)
        self.canvas._tkcanvas.pack(side='top', fill='both', expand=1)

        if not varlabels:
            varlabels = result.var_names
        if best_vals and isinstance(best_vals, bool):
            best_vals = result.params

        for i, par1 in enumerate(result.var_names):
            for j, par2 in enumerate(result.var_names):

                # parameter vs chi2 in case of only one parameter
                if npars == 1:
                    axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                    axes.set_ylabel(r'$\chi^{2}$')
                    axes.set_xlabel(varlabels[i])
                    if best_vals:
                        axes.axvline(best_vals[par1].value, ls='dashed', color='r')

                # parameter vs chi2 profile on top
                elif i == j and j < npars - 1:
                    if i == 0:
                        axes[0, 0].axis('off')
                    ax = axes[i, j + 1]
                    red_axis = tuple([a for a in range(npars) if a != i])
                    ax.plot(np.unique(result.brute_grid[i]),
                            np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            'o', ms=3)
                    ax.set_ylabel(r'$\chi^{2}$')
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.set_ticks_position('right')
                    ax.set_xticks([])
                    if best_vals:
                        ax.axvline(best_vals[par1].value, ls='dashed', color='r')

                # parameter vs chi2 profile on the left
                elif j == 0 and i > 0:
                    ax = axes[i, j]
                    red_axis = tuple([a for a in range(npars) if a != i])
                    ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            np.unique(result.brute_grid[i]), 'o', ms=3)
                    ax.invert_xaxis()
                    ax.set_ylabel(varlabels[i])
                    if i != npars - 1:
                        ax.set_xticks([])
                    elif i == npars - 1:
                        ax.set_xlabel(r'$\chi^{2}$')
                    if best_vals:
                        ax.axhline(best_vals[par1].value, ls='dashed', color='r')

                # contour plots for all combinations of two parameters
                elif j > i:
                    ax = axes[j, i + 1]
                    red_axis = tuple([a for a in range(npars) if a != i and a != j])
                    X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                       np.unique(result.brute_grid[j]))
                    lvls1 = np.linspace(result.brute_Jout.min(),
                                        np.median(result.brute_Jout) / 2.0, 7, dtype='int')
                    lvls2 = np.linspace(np.median(result.brute_Jout) / 2.0,
                                        np.median(result.brute_Jout), 3, dtype='int')
                    lvls = np.unique(np.concatenate((lvls1, lvls2)))
                    ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                                lvls, norm=LogNorm())
                    ax.set_yticks([])
                    if best_vals:
                        ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                        ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                        ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                    if j != npars - 1:
                        ax.set_xticks([])
                    elif j == npars - 1:
                        ax.set_xlabel(varlabels[i])
                    if j - i >= 2:
                        axes[i, j].axis('off')

        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

        if output is not None:
            self.fig.savefig(output)

