import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from .guiForFitOperation import guiForFitOperation
from core import Experiment

from tkinter import filedialog, messagebox, simpledialog

from .Dialog import fitIRFDialog, generateIRFDialog




class guiForFitOperation_Lifetime(guiForFitOperation):

    def __init__(self, master_frame, controller, model_names, nb_param_fit, is_burst_analysis=False):
        super().__init__(master_frame, controller, model_names, nb_param_fit, fitModeName="lifetime", is_burst_analysis=is_burst_analysis)

        # TODO
        # ttk.Button(self.cmd_frame, text="Estimate nb of exp", command=self.estimate_nb_of_exp).grid(row=9, column=0)


    # def changeModel(self, event):
    #     # Methode virtuelle, voir les classes dérivées.
    #     model_name = self.cb_model_sv.get()
    #
    #     self.measurement.set_model(model_name)
    #     self.create_gui_from_measurement_params()
    #     self.set_fit_formula(self.measurement.model.fit_formula)
    #
    #
    #     # self.set_fit_formula(r"\frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})")
    #
    #     # else:
    #     #     for i in range(nb_fit_param) :
    #     #         self.list_label_string_variable_fit[i].set("")
    #     #         self.enable_disable_ui(0)

    # def copy_param_from_fit(self):
    #     # self.measurement.fit_results.params
    #     # TODO
    #     pass

    def estimate_nb_of_exp(self):
        params = self.get_fit_params()
        result = self.measurement.estimate_nb_of_exp(params)
        # self.plot_results_brute(result)


class lifeTimeAnalyze_gui():
    def __init__(self, masterFrame, controller, appearenceParam, measurement, is_burst_analysis=False):
        self.master_frame = masterFrame
        self.controller = controller
        self.appearenceParam = appearenceParam
        self.measurement = measurement
        self.is_keep_selection_for_filter = True
        self.is_burst_analysis = is_burst_analysis
        self.type = "lifetime"
        # self.is_graph_x_ns = True

    def populate(self):
        self.frameMicro_graph = tk.LabelFrame(self.master_frame, text="Graph",
                                              borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_filter = tk.LabelFrame(self.master_frame, text="Filter",
                                               borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_IR_signal = tk.LabelFrame(self.master_frame, text="Static param fit",
                                                  borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_IR = tk.LabelFrame(self.frameMicro_IR_signal, text="IR",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_Signal = tk.LabelFrame(self.frameMicro_IR_signal, text="Signal",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_fit = tk.LabelFrame(self.master_frame, text="Fit",
                                            borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.frameMicro_graph.pack(side="left", fill="both", expand=True)
        self.frameMicro_filter.pack(side="left", fill="both", expand=True)
        self.frameMicro_IR_signal.pack(side="left", fill="both", expand=True)
        self.frameMicro_IR.pack(side="top", fill="both", expand=True)
        self.frameMicro_Signal.pack(side="top", fill="both", expand=True)
        self.frameMicro_fit.pack(side="left", fill="both", expand=True)


        #Graph
        ttk.Button(self.frameMicro_graph, text="Graph", width=6, command=self.launch_micro_time_histo).grid(row=0, column=0)


        self.is_semi_log = tk.IntVar()
        ttk.Checkbutton(self.frameMicro_graph, text="SemiLog ?", variable=self.is_semi_log, command=self.change_semi_log_graph).grid(row=0, column=1)
        self.is_semi_log.set(1)

        # self.toggle_button_graph = ttk.Button(self.frameMicro_graph, text="ns", width=15, command=self.toggle_graph_x)
        # self.toggle_button_graph.grid(row=0, column=0)

        ttk.Label(self.frameMicro_graph, text='channel :').grid(row=1, column=0)
        self.num_channel_sv = tk.StringVar()
        e = ttk.Entry(self.frameMicro_graph, textvariable=self.num_channel_sv, justify=tk.CENTER, width=7)
        e.grid(row=1, column=1)
        self.num_channel_sv.set('0')

        self.is_overlay_on_time_zoom = tk.IntVar()
        ttk.Checkbutton(self.frameMicro_graph, text="Overlay on time zoom", variable=self.is_overlay_on_time_zoom, command=self.update_navigation).grid(row=2, column=0, columnspan=2)

        ttk.Label(self.frameMicro_graph, text='ymin for log graph').grid(row=3, column=0)
        self.low_y_log_graph_sv = tk.StringVar()
        e = ttk.Entry(self.frameMicro_graph, textvariable=self.low_y_log_graph_sv, justify=tk.CENTER, width=7)
        e.grid(row=3, column=1)
        e.bind('<Return>', self.change_low_y_log_graph)
        self.low_y_log_graph_sv.set('10')

        #Filter
        self.toggle_button = ttk.Button(self.frameMicro_filter, text="Keep selection", width=15, command=self.toggle_filter_mode)
        self.toggle_button.grid(row=0, column=0)

        ttk.Button(self.frameMicro_filter, text="Filter", width=6, command=self.microtime_filter).grid(row=1, column=0)

        #IR
        tk.Label(self.frameMicro_IR, text="Name :").grid(row=0, column=0)

        self.cb_IRF_sv = tk.StringVar()
        self.cb_IRF = ttk.Combobox(self.frameMicro_IR, width=20, justify=tk.CENTER, textvariable=self.cb_IRF_sv,
                          values='', state='readonly')
        self.cb_IRF.bind('<<ComboboxSelected>>', self.select_IR)
        irf_name_list = self.controller.model.get_irf_name_list()
        irf_name_list.insert(0, "None")
        self.cb_IRF['values'] = irf_name_list
        self.cb_IRF_sv.set(irf_name_list[0])
        self.cb_IRF.set(irf_name_list[0])
        self.cb_IRF.grid(row=0, column=1)


        ttk.Button(self.frameMicro_IR, text="Add new IRF",  command=self.openIR_file).grid(row=0, column=2)
        ttk.Button(self.frameMicro_IR, text="Generate",  command=self.generateIR_file).grid(row=0, column=3)

        self.isDraw_IR = tk.IntVar()
        ttk.Checkbutton(self.frameMicro_IR, text="Use IR ?", variable=self.isDraw_IR, command=self.is_use_IR).grid(row=1, column=0)

        # tk.Label(self.frameMicro_IR, text="IR Name :").grid(row=1, column=0)

        # self.ir_name_sv = tk.StringVar()
        # ttk.Entry(self.frameMicro_IR, textvariable=self.ir_name_sv, justify=tk.CENTER, width=50).grid(row=1, column=1, columnspan=2)
        # self.ir_name_sv.set("None loaded yet")

        tk.Label(self.frameMicro_IR, text="Beginning % :").grid(row=2, column=0)

        self.ir_start_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.ir_start_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        self.ir_start_sv.set(0)
        w.grid(row=2, column=1)

        tk.Label(self.frameMicro_IR, text="End % :").grid(row=3, column=0)

        self.ir_end_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.ir_end_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        self.ir_end_sv.set(100)
        w.grid(row=3, column=1)

        # tk.Label(self.frameMicro_IR, text="Shift (µchannel) :").grid(row=4, column=0)
        # self.shiftIR_amount_sv = tk.StringVar()
        # w = ttk.Entry(self.frameMicro_IR, textvariable=self.shiftIR_amount_sv, justify=tk.CENTER, width=15)
        # w.bind('<Return>', self.change_IR)
        # w.grid(row=4, column=1)
        # self.shiftIR_amount_sv.set(0)

        tk.Label(self.frameMicro_IR, text="Background").grid(row=5, column=0)
        self.bckg_IR_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.bckg_IR_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        w.grid(row=5, column=1)
        self.bckg_IR_sv.set(0)

        ttk.Button(self.frameMicro_IR, text="Auto",  command=self.auto_bckgnd).grid(row=3, column=2)

        # b = ttk.Button(self.frameMicro_IR, text="Fit IR",  command=self.fit_IR).grid(row=4, column=0)

        #Signal
        # tk.Label(self.frameMicro_Signal, text="Signal bckgnd :").grid(row=0, column=0)
        # self.signal_bckgnd_sv = tk.StringVar()
        # self.signal_bckgnd_sv.set("0")
        # ttk.Entry(self.frameMicro_Signal, textvariable=self.signal_bckgnd_sv, justify=tk.CENTER, width=15).grid(row=0, column=1)
        # ttk.Button(self.frameMicro_Signal, text="use cursor", command=self.select_bckgnd_w_cursor).grid(row=0, column=2)

        #FIT
        # model_names -> cf core/lifetime.py
        self.gui_for_fit_operation = guiForFitOperation_Lifetime(self.frameMicro_fit, self.controller,
                                                                 model_names=('One Decay IRF', 'One Decay Tail', 'Two Decays IRF', 'Two Decays IRF A1 A2', 'Two Decays Tail', 'Two Decays Tail A1 A2', 'IRF Becker', "MonoExp for IRF", "Three Decays Tail A1 A2 A3"), nb_param_fit=8,
                                                                 is_burst_analysis=self.is_burst_analysis)
        self.gui_for_fit_operation.populate()

    def openIR_file(self):
        file_path = filedialog.askopenfilename(title="Open IR File", initialdir=self.controller.view.saveDir)
        if file_path == None or file_path == '':
            return None
        file_name = self.controller.open_and_set_IRF_file(file_path)

        if file_name not in self.cb_IRF['values']:
            # Remove
            self.cb_IRF['values'] += (file_name,)
            self.cb_IRF.set(file_name)
            self.select_IR(None)

        self.controller.update_analyze()

    def select_IR(self, event):
        if self.cb_IRF.get() != "None":
            self.controller.set_IRF(self.cb_IRF.get())

    def is_use_IR(self):
        self.change_IR(None)

    def select_bckgnd_w_cursor(self):
        pass

    def update_navigation(self):
        self.controller.update_navigation()

    def update_analyze(self):
        self.controller.update_analyze()

    def fit_IR(self):
        d = fitIRFDialog(self.master_frame, title="Initial fit parameters")
        if d.result is not None:
            iniParams = d.result
            self.controller.fit_IR(iniParams)

    def generateIR_file(self):
        d = generateIRFDialog(self.master_frame, title="Generate IRF")
        if d.result is not None:
            params_dict = d.result
            file_name = self.controller.generate_IRF(params_dict)

            if file_name not in self.cb_IRF['values']:
                # Remove
                self.cb_IRF['values'] += (file_name,)
                self.cb_IRF.set(file_name)
                self.select_IR(None)

            self.controller.update_analyze()

    def change_low_y_log_graph(self, e):
        low_y_log_graph = int(self.low_y_log_graph_sv.get())
        self.controller.graph_results.low_limit_log_lifetime = low_y_log_graph
        self.launch_micro_time_histo()


    def auto_bckgnd(self):
        pass

    def change_IR(self, e):
        start = float(self.ir_start_sv.get())
        if start < 0:
            start = 0
        elif start > 100:
            start = 100
        self.controller.current_measurement.IRF.start = start

        end = float(self.ir_end_sv.get())
        if end < 0:
            end = 0
        elif end > 100:
            end = 100
        self.controller.current_measurement.IRF.end = end

        # self.controller.current_measurement.IR_shift = float(self.shiftIR_amount_sv.get())
        self.controller.current_measurement.set_use_IR(self.isDraw_IR.get())

        self.controller.current_measurement.IRF.bckgnd = int(self.bckg_IR_sv.get())

        if self.controller.current_measurement.IRF.process() == "OK":
            self.controller.update_analyze()

    def launch_micro_time_histo(self):
        #FIXME
        low_y_log_graph = int(self.low_y_log_graph_sv.get())
        self.controller.graph_results.low_limit_log_lifetime = low_y_log_graph
        self.controller.calculate_measurement()
        self.controller.graph_measurement()
        # self.mainGUI.controller.drawMicroTimeHisto(self.mainGUI.currentChannel, self.mainGUI.currentTimeWindow[0], self.mainGUI.currentTimeWindow[1])
        # print("launchMicroTimeHisto")

    def change_semi_log_graph(self):
        self.measurement.is_plot_log = self.is_semi_log.get()

    def toggle_filter_mode(self):
        if self.is_keep_selection_for_filter:
            self.toggle_button.config(text='Filter selection')
            self.is_keep_selection_for_filter = False
        else:
            self.toggle_button.config(text='Keep selection')
            self.is_keep_selection_for_filter = True

    # def toggle_graph_x(self):
    #     if self.is_graph_x_ns:
    #         self.toggle_button_graph.config(text='num channel')
    #         self.is_graph_x_ns = False
    #     else:
    #         self.toggle_button.config(text='ns')
    #         self.is_graph_x_ns = True

    def microtime_filter(self):
        num_channel = int(self.num_channel_sv.get())
        self.controller.microtime_filter(num_channel, self.is_keep_selection_for_filter)


