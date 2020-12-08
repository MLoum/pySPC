import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from .guiForFitOperation import guiForFitOperation
from core import Experiment

from tkinter import filedialog, messagebox, simpledialog

from .Dialog import fitIRFDialog


class guiForFitOperation_Phospho(guiForFitOperation):

    def __init__(self, master_frame, controller, model_names, nb_param_fit, is_burst_analysis=False):
        super().__init__(master_frame, controller, model_names, nb_param_fit, fitModeName="lifetime", is_burst_analysis=is_burst_analysis)

    # def change_model(self, event):
    #     # Methode virtuelle, voir les classes dérivées.
    #     nbFitParam = self.nb_max_param_fit
    #
    #     if self.cb_model_sv.get() == "One Decay Tail":
    #         self.list_label_string_variable_fit[0].set("t0")
    #         self.list_label_string_variable_fit[1].set("amp")
    #         self.list_label_string_variable_fit[2].set("tau")
    #         self.list_label_string_variable_fit[3].set("cst")
    #         self.enable_disable_ui(4)
    #         self.set_fit_formula(r"cst + e^{-(t-t_0)/\tau}")
    #
    #     elif self.cb_model_sv.get() == "Two Decays Tail":
    #         self.list_label_string_variable_fit[0].set("tau1")
    #         self.list_label_string_variable_fit[1].set("a1")
    #         self.list_label_string_variable_fit[2].set("tau2")
    #         self.list_label_string_variable_fit[3].set("bckgnd")
    #         self.enable_disable_ui(5)
    #         self.set_fit_formula(r" a1 . e^{-t/\tau_1} + (1-a_1) \times e^{-t/\tau_2}) + bckgnd")
    #
    #         self.list_entry_string_variable_fit_min[0].set("0")
    #         self.list_entry_string_variable_fit_min[1].set("0")
    #         self.list_entry_string_variable_fit_min[2].set("0")
    #         self.list_entry_string_variable_fit_min[4].set("0")
    #
    #     else:
    #         for i in range(nbFitParam) :
    #             self.list_label_string_variable_fit[i].set("")
    #             self.enable_disable_ui(0)



class PhosphoAnalyze_gui():
    def __init__(self, masterFrame, controller, appearenceParam, measurement, is_burst_analysis=False):
        self.masterFrame = masterFrame
        self.controller = controller
        self.appearenceParam = appearenceParam
        self.measurement = measurement
        self.is_keep_selection_for_filter = True
        self.is_burst_analysis = is_burst_analysis
        self.type = "lifetime"
        # self.is_graph_x_ns = True

    def populate(self):
        self.frame_graph = tk.LabelFrame(self.masterFrame, text="Graph",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        # self.frameMicro_filter = tk.LabelFrame(self.masterFrame, text="Filter",
        #                                  borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frame_fit = tk.LabelFrame(self.masterFrame, text="Fit",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.frame_graph.pack(side="left", fill="both", expand=True)
        self.frame_fit.pack(side="left", fill="both", expand=True)

        #Graph
        ttk.Label(self.frame_graph, text='channel Start :').grid(row=0, column=0)
        self.num_channel_start_sv = tk.StringVar(value="1") # 1 based
        ttk.Entry(self.frame_graph, textvariable=self.num_channel_start_sv, justify=tk.CENTER, width=7).grid(row=0, column=1)

        ttk.Label(self.frame_graph, text='channel Stop :').grid(row=0, column=2)
        self.num_channel_stop_sv = tk.StringVar(value="2")
        ttk.Entry(self.frame_graph, textvariable=self.num_channel_stop_sv, justify=tk.CENTER, width=7).grid(row=0, column=3)


        ttk.Label(self.frame_graph, text='Time step (µs) :').grid(row=1, column=0)
        self.time_step_micros_sv = tk.StringVar(value="10")
        ttk.Entry(self.frame_graph, textvariable=self.time_step_micros_sv, justify=tk.CENTER, width=7).grid(row=1, column=1)

        ttk.Label(self.frame_graph, text='Start time (µs) :').grid(row=2, column=0)
        self.min_time_micros_sv = tk.StringVar(value="1")
        ttk.Entry(self.frame_graph, textvariable=self.min_time_micros_sv, justify=tk.CENTER, width=7).grid(row=2, column=1)


        ttk.Label(self.frame_graph, text='Max Time (ms) :').grid(row=2, column=2)
        self.max_time_ms_sv = tk.StringVar(value="5000")
        ttk.Entry(self.frame_graph, textvariable=self.max_time_ms_sv, justify=tk.CENTER, width=7).grid(row=2, column=3)

        self.is_semi_log = tk.IntVar()
        ttk.Checkbutton(self.frame_graph, text="SemiLog ?", variable=self.is_semi_log, command=self.update_analyze).grid(row=3, column=0)


        ttk.Button(self.frame_graph, text="Graph", width=12, command=self.launch_phospho_histo).grid(row=4, column=0, columnspan=3)



        #FIT
        # model_names -> cf core/lifetime.py
        self.gui_for_fit_operation = guiForFitOperation_Phospho(self.frame_fit, self.controller,
                                                                 model_names=('One Decay Tail', 'Two Decays Tail'), nb_param_fit=8,
                                                                 is_burst_analysis=self.is_burst_analysis)
        self.gui_for_fit_operation.populate()


    def update_navigation(self):
        self.controller.update_navigation()

    def update_analyze(self):
        self.controller.update_analyze()

    def launch_phospho_histo(self):
        #FIXME
        self.controller.calculate_measurement()
        self.controller.graph_measurement()


