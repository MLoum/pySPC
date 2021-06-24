
# TODO highlighted burst
# TODO not dans les filter
# TODO faire les mesures de temps de vie avec les burst et les fit auto (gros morceau).


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from .analyze_Lifetime import lifeTimeAnalyze_gui
from .graph.interactiveGraphs import InteractiveGraph
import numpy as np
from scipy.special import factorial
from scipy.stats import poisson
import matplotlib.patches as patches

class BurstAnalysis_gui():
    def __init__(self, master_frame, controller, appearence_param, measurement=None):
        self.master_frame = master_frame
        self.controller = controller
        self.appearence_param = appearence_param
        self.burst_measure = measurement
        self.current_burst = None
        self.modelNames = ("Bin and Threshold", "todo")
        self.measurement_iid_dict = {}
        self.burst_iid_dict = {}
        self.populate()


    def populate(self):
        self.top_level = tk.Toplevel(self.master_frame)
        self.top_level.title("Burst Detection and analysis")

        self.notebook = ttk.Notebook(self.top_level)
        self.notebook.pack(expand=True, fill="both")

        self.frame_detection = tk.Frame(self.notebook)
        self.frame_detection.pack(side="top", fill="both", expand=True)
        self.notebook.add(self.frame_detection, text='Burst Detection')


        # label = ttk.Label(self.frame_detection, text='Model :')
        # label.grid(row=0, column=0)
        #
        # self.combo_box_model_sv = tk.StringVar()
        # cb = ttk.Combobox(self.frame_detection, width=15, justify=tk.CENTER, textvariable=self.combo_box_model_sv,
        #                   values='', state='readonly')
        # cb.bind('<<ComboboxSelected>>', self.change_model)
        # cb['values'] = self.modelNames
        # self.combo_box_model_sv.set("Bin and Threshold")
        # cb.set(self.modelNames[0])
        # cb.grid(row=0, column=1)


        self.frame_chronogram = tk.LabelFrame(self.frame_detection, text="a) chronogramm", borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_chronogram.pack(side="top", fill="both", expand=True)

        ttk.Label(self.frame_chronogram, text='bin size (ms)').grid(row=0, column=0)
        self.binsize_sv = tk.StringVar()
        e = ttk.Entry(self.frame_chronogram, textvariable=self.binsize_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=1)
        self.binsize_sv.set("10")

        b = ttk.Button(self.frame_chronogram, text="bin signal", command=self.bin_signal)
        b.grid(row=0, column=2)

        # Threshold
        #############

        self.frame_threshold = tk.LabelFrame(self.frame_detection, text="b) threshold",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_threshold.pack(side="top", fill="both", expand=True)

        label = ttk.Label(self.frame_threshold, text='Threshold Burst')
        label.grid(row=0, column=0)

        self.threshold_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_threshold, textvariable=self.threshold_burst_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=1)

        ttk.Label(self.frame_threshold, text='Threshold Flank').grid(row=0, column=2)
        self.threshold_flank_sv = tk.StringVar()
        e = ttk.Entry(self.frame_threshold, textvariable=self.threshold_flank_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=3)

        ttk.Label(self.frame_threshold, text='% False Negative').grid(row=0, column=4)

        self.false_negative_sv = tk.StringVar()
        e = ttk.Entry(self.frame_threshold, textvariable=self.false_negative_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=5)
        self.false_negative_sv.set("1")

        ttk.Button(self.frame_threshold, text="Auto Threshold", command=self.auto_threshold).grid(row=0, column=6)

        self.frame_graph_trh_grid = tk.Frame(self.frame_threshold)
        self.frame_graph_trh_grid.grid(row=1, column=0, columnspan=7)

        self.frame_graph_threshold = tk.Frame(self.frame_graph_trh_grid)
        self.frame_graph_threshold.pack(side="top", fill="both", expand=True)

        self.pch_graph = Graph_PCH_burst(self.frame_graph_threshold, self.controller.view, self.controller, self, figsize=(5,2), dpi=75)

        # Burst Param
        #################

        self.frame_burst_param = tk.LabelFrame(self.frame_detection, text="c) burst parameters",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_burst_param.pack(side="top", fill="both", expand=True)

        ttk.Label(self.frame_burst_param, text='Minimum nb of successive bins').grid(row=0, column=0)
        self.min_succesive_bin_sv = tk.StringVar()
        e = ttk.Entry(self.frame_burst_param, textvariable=self.min_succesive_bin_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=1)
        self.min_succesive_bin_sv.set("2")

        ttk.Label(self.frame_burst_param, text='Maxixum nb of successive NOISE bins').grid(row=0, column=2)
        self.max_succesive_noise_bin_sv = tk.StringVar()
        e = ttk.Entry(self.frame_burst_param, textvariable=self.max_succesive_noise_bin_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=3)
        self.max_succesive_noise_bin_sv.set("1")

        ttk.Label(self.frame_burst_param, text='Min nb of photons in burst').grid(row=0, column=4)
        self.min_nb_photon_sv = tk.StringVar()
        e = ttk.Entry(self.frame_burst_param, textvariable=self.min_nb_photon_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=5)
        self.min_nb_photon_sv.set("100")

        ttk.Button(self.frame_burst_param, text="Launch Detection", command=self.launch_detection).grid(row=0, column=6)

        self.frame_operation = tk.LabelFrame(self.frame_detection, text="d) What to do",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_operation.pack(side="top", fill="both", expand=True)

        # Statistics
        ##############

        self.frame_statistics = tk.LabelFrame(self.frame_detection, text="e) Statistics information",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_statistics.pack(side="top", fill="both", expand=True)

        ttk.Label(self.frame_statistics, text='nb of burst').grid(row=0, column=0)
        self.nb_of_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_statistics, textvariable=self.nb_of_burst_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=1)
        self.nb_of_burst_sv.set("0")

        ttk.Label(self.frame_statistics, text='nb of rejected short burst').grid(row=0, column=2)
        self.nb_of_short_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_statistics, textvariable=self.nb_of_short_burst_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=3)
        self.nb_of_short_burst_sv.set("0")

        pad_graph = 7

        # Noise
        #TODO cf core.

        # self.frame_graph_Noise_CPS_grid = tk.Frame(self.frame_statistics)
        # self.frame_graph_Noise_CPS_grid.grid(row=1, column=0, padx=pad_graph)
        #
        # self.frame_graph_Noise_CPS = tk.Frame(self.frame_graph_Noise_CPS_grid)
        # self.frame_graph_Noise_CPS.pack(side="top", fill="both", expand=True)
        #
        # self.noise_CPS_graph = Graph_stat(self.frame_graph_Noise_CPS, self.controller.view, self.controller, self, figsize=(2,2), dpi=75)
        #
        # self.frame_graph_Noise_duration_grid = tk.Frame(self.frame_statistics)
        # self.frame_graph_Noise_duration_grid.grid(row=1, column=1, padx=pad_graph)
        #
        # self.frame_graph_Noise_duration = tk.Frame(self.frame_graph_Noise_duration_grid)
        # self.frame_graph_Noise_duration.pack(side="top", fill="both", expand=True)
        #
        # self.noise_duration_graph = Graph_stat(self.frame_graph_Noise_duration, self.controller.view, self.controller, self, figsize=(2,2), dpi=75)

        # Burst

        self.frame_graph_Burst_int_grid = tk.Frame(self.frame_statistics)
        self.frame_graph_Burst_int_grid.grid(row=1, column=2, padx=pad_graph)

        self.frame_graph_Burst_int = tk.Frame(self.frame_graph_Burst_int_grid)
        self.frame_graph_Burst_int.pack(side="top", fill="both", expand=True)

        self.burst_int_graph = Graph_stat(self.frame_graph_Burst_int, self.controller.view, self.controller, self, figsize=(2,2), dpi=75)

        self.frame_graph_Burst_duration_grid = tk.Frame(self.frame_statistics)
        self.frame_graph_Burst_duration_grid.grid(row=1, column=3, padx=pad_graph)

        self.frame_graph_Burst_duration = tk.Frame(self.frame_graph_Burst_duration_grid)
        self.frame_graph_Burst_duration.pack(side="top", fill="both", expand=True)

        self.burst_duration_graph = Graph_stat(self.frame_graph_Burst_duration, self.controller.view, self.controller, self,
                                          figsize=(2, 2), dpi=75)

        self.frame_graph_Burst_CPS_grid = tk.Frame(self.frame_statistics)
        self.frame_graph_Burst_CPS_grid.grid(row=1, column=4, padx=pad_graph)

        self.frame_graph_Burst_CPS = tk.Frame(self.frame_graph_Burst_CPS_grid)
        self.frame_graph_Burst_CPS.pack(side="top", fill="both", expand=True)

        self.burst_CPS_graph = Graph_stat(self.frame_graph_Burst_CPS, self.controller.view, self.controller, self,
                                          figsize=(2, 2), dpi=75)

        # tests on burst
        ##############

        self.frame_test = tk.LabelFrame(self.frame_detection, text="f) Tests before validation",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_test.pack(side="top", fill="both", expand=True)


        ttk.Label(self.frame_test, text='Num burst').grid(row=0, column=0)
        self.num_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_test, textvariable=self.num_burst_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=1)
        self.num_burst_sv.set("0")

        ttk.Button(self.frame_test, text="previous burst", command=self.previous_burst).grid(row=1, column=0)
        ttk.Button(self.frame_test, text="next burst", command=self.next_burst).grid(row=1, column=1)


        ttk.Label(self.frame_test, text='Nb photon').grid(row=2, column=0)
        self.nb_photon_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_test, textvariable=self.nb_photon_burst_sv, justify=tk.CENTER, width=12)
        e.config(state=tk.DISABLED)
        e.grid(row=2, column=1)

        ttk.Label(self.frame_test, text='length (µs)').grid(row=2, column=2)
        self.length_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_test, textvariable=self.nb_photon_burst_sv, justify=tk.CENTER, width=12)
        e.config(state=tk.DISABLED)
        e.grid(row=2, column=3)

        ttk.Label(self.frame_test, text='CPS').grid(row=2, column=4)
        self.cps_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_test, textvariable=self.cps_burst_sv, justify=tk.CENTER, width=12)
        e.config(state=tk.DISABLED)
        e.grid(row=2, column=5)

        #graph chrono

        #graph results

        # Validate
        #####################

        self.frame_validate = tk.LabelFrame(self.frame_detection, text="g) validate",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_validate.pack(side="top", fill="both", expand=True)
        ttk.Button(self.frame_validate, text="validate", command=self.validate).grid(row=1, column=1)

        # self.createCallBacks()
        # self.createWidgets()


        # FRAME ANALYSIS
        self.frame_filter = tk.Frame(self.notebook)
        self.frame_filter.pack(side="top", fill="both", expand=True)
        self.notebook.add(self.frame_filter, text='Burst Analysis')

        #burst list
        self.frame_burst_list = tk.LabelFrame(self.frame_filter, text="Burst List",
                                              borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_burst_list.pack(side="top", fill="both", expand=True)

        #https://riptutorial.com/tkinter/example/31885/customize-a-treeview
        self.tree_view = ttk.Treeview(self.frame_burst_list)
        self.tree_view["columns"] = ("name", "num burst", "tick start", "duration", "nb photon", "CPS", "channel", "m. type", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")
        # remove first empty column with the identifier
        # self.tree_view['show'] = 'headings'
        # tree.column("#0", width=270, minwidth=270, stretch=tk.NO) tree.column("one", width=150, minwidth=150, stretch=tk.NO) tree.column("two", width=400, minwidth=200) tree.column("three", width=80, minwidth=50, stretch=tk.NO)
        self.tree_view.column("#0", width=25, stretch=tk.NO)
        self.tree_view.column("name", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("num burst", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("tick start", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("duration", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("nb photon", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("CPS", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("channel", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("m. type", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("p1", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("p2", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("p3", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("p4", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("p5", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("p6", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("p7", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("p8", width=50, stretch=tk.YES, anchor=tk.CENTER)

        self.tree_view.heading("name", text="name")
        self.tree_view.heading("num burst", text="num burst")
        self.tree_view.heading("tick start", text="tick start")
        self.tree_view.heading("duration", text="duration")
        self.tree_view.heading("nb photon", text="nb photon")
        self.tree_view.heading("CPS", text="CPS")
        self.tree_view.heading("channel", text="channel")
        self.tree_view.heading("m. type", text="m. type")
        self.tree_view.heading("p1", text="p1")
        self.tree_view.heading("p2", text="p2")
        self.tree_view.heading("p3", text="p3")
        self.tree_view.heading("p4", text="p4")
        self.tree_view.heading("p5", text="p5")
        self.tree_view.heading("p6", text="p6")
        self.tree_view.heading("p7", text="p7")
        self.tree_view.heading("p8", text="p8")

        #FIXME only change text color to light gray
        self.tree_view.tag_configure('filtered', foreground='gray50')
        self.tree_view.tag_configure('filtered', background='gray20')

        self.tree_view.tag_configure('highlighted', background='gray90')
        self.tree_view.tag_configure('highlighted', foreground='gold3')

        ysb = ttk.Scrollbar(self.frame_burst_list, orient='vertical', command=self.tree_view.yview)
        self.tree_view.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        self.tree_view.configure(yscroll=ysb.set)

        self.tree_view.bind('<<TreeviewSelect>>', self.treeview_measurement_select)
        self.tree_view.bind("<Double-1>", self.on_double_click_treeview)

        self.frame_tree_params = tk.Frame(self.frame_burst_list)
        self.frame_tree_params.grid(row=0, column=2, sticky='nsew')

        self.check_show_filtered_iv = tk.IntVar(value=0)
        # tk.Checkbutton(self.frame_tree_params, text='show Filtered as gray', variable=self.check_show_filtered_iv, command=self.update_ui(), onvalue=1, offvalue=0).grid(row=0, column=0, columnspan=2)

        ttk.Label(self.frame_tree_params, text='Nb filtered burst').grid(row=1, column=0)
        self.nb_filtered_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_tree_params, textvariable=self.nb_filtered_burst_sv, justify=tk.CENTER, width=12)
        e.config(state=tk.DISABLED)
        e.grid(row=1, column=1)
        ttk.Label(self.frame_tree_params, text='/').grid(row=1, column=2)
        self.nb_total_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_tree_params, textvariable=self.nb_total_burst_sv, justify=tk.CENTER, width=12)
        e.config(state=tk.DISABLED)
        e.grid(row=1, column=3)

        ttk.Button(self.frame_tree_params, text="Toggle Filtered", command=self.toggle_filter_burst).grid(row=2, column=0)
        ttk.Button(self.frame_tree_params, text="Toggle Highlighted", command=self.toggle_highlight_burst).grid(row=3,
                                                                                                          column=0)





        # Filter burst
        self.frame_filter_burst = tk.LabelFrame(self.frame_filter, text="Filter Burst",
                                                borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_filter_burst.pack(side="top", fill="both", expand=True)

        pad_filter = 5

        ttk.Label(self.frame_filter_burst, text='           Filter 1           ', background="white").grid(row=0, column=0,  columnspan=6, padx=pad_filter)
        ttk.Label(self.frame_filter_burst, text='           Filter 2           ', background="white").grid(row=0, column=7,  columnspan=6, padx=pad_filter)

        self.is_not_f1 = True
        self.button_not_f1 = tk.Button(self.frame_filter_burst, text="not", width=3, command=self.toggle_not_f1)
        self.button_not_f1.grid(row=1, column=0, padx=pad_filter)

        # So that not start as false
        self.toggle_not_f1()


        self.filter_1_low_sv = tk.StringVar()
        ttk.Entry(self.frame_filter_burst, textvariable=self.filter_1_low_sv, justify=tk.CENTER, width=12).grid(row=1, column=1, padx=pad_filter)
        ttk.Label(self.frame_filter_burst, text=' < ').grid(row=1, column=2, padx=pad_filter)

        self.cb_value_filter_1_sv = tk.StringVar()
        self.cb_value_filter_1 = ttk.Combobox(self.frame_filter_burst, width=25, justify=tk.CENTER, textvariable=self.cb_value_filter_1_sv, values='')
        self.cb_value_filter_1['values'] = ["None", "duration", "nb_photon", "CPS", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
        self.cb_value_filter_1.set('None')
        self.cb_value_filter_1.bind('<<ComboboxSelected>>', self.change_filter1_type)
        self.cb_value_filter_1.grid(row=1, column=3, padx=pad_filter)

        ttk.Label(self.frame_filter_burst, text=' < ').grid(row=1, column=4, padx=pad_filter)
        self.filter_1_high_sv = tk.StringVar()
        ttk.Entry(self.frame_filter_burst, textvariable=self.filter_1_high_sv, justify=tk.CENTER, width=12).grid(row=1, column=5, padx=pad_filter)

        self.cb_value_filter_bool_op_sv = tk.StringVar()
        self.cb_value_filter_bool_op = ttk.Combobox(self.frame_filter_burst, width=25, justify=tk.CENTER, textvariable=self.cb_value_filter_bool_op_sv, values='')
        self.cb_value_filter_bool_op['values'] = ["and", "or", "xor"]
        self.cb_value_filter_bool_op.set('or')

        self.cb_value_filter_bool_op.grid(row=1, column=6, padx=pad_filter)

        self.is_not_f2 = True
        self.button_not_f2 = tk.Button(self.frame_filter_burst, text="not", width=3, command=self.toggle_not_f2)
        self.button_not_f2.grid(row=1, column=7, padx=pad_filter)
        # So that not start as false
        self.toggle_not_f2()




        self.filter_2_low_sv = tk.StringVar()
        ttk.Entry(self.frame_filter_burst, textvariable=self.filter_2_low_sv, justify=tk.CENTER, width=12).grid(row=1, column=8, padx=pad_filter)
        ttk.Label(self.frame_filter_burst, text=' < ').grid(row=1, column=9, padx=pad_filter)

        self.cb_value_filter_2_sv = tk.StringVar()
        self.cb_value_filter_2 = ttk.Combobox(self.frame_filter_burst, width=25, justify=tk.CENTER, textvariable=self.cb_value_filter_2_sv, values='')
        self.cb_value_filter_2['values'] = ["None", "duration", "nb_photon", "CPS", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
        self.cb_value_filter_2.set('None')
        self.cb_value_filter_2.bind('<<ComboboxSelected>>', self.change_filter2_type)
        self.cb_value_filter_2.grid(row=1, column=10, padx=pad_filter)

        ttk.Label(self.frame_filter_burst, text=' < ').grid(row=1, column=11, padx=pad_filter)

        self.filter_2_high_sv = tk.StringVar()
        ttk.Entry(self.frame_filter_burst, textvariable=self.filter_2_high_sv, justify=tk.CENTER, width=12).grid(row=1, column=12, padx=pad_filter)


        self.frame_vs_plot_f1_grid = tk.Frame(self.frame_filter_burst)
        self.frame_vs_plot_f1_grid.grid(row=2, column=0, columnspan=6)

        self.frame_vs_plot_f1 = tk.Frame(self.frame_vs_plot_f1_grid)
        self.frame_vs_plot_f1.pack(side="top", fill="both", expand=True)

        self.filter_1_graph = Graph_filter(self.frame_vs_plot_f1, self.controller.view, self.controller, self.filter_1_low_sv, self.filter_1_high_sv, figsize=(7,2), dpi=75)

        self.frame_vs_plot_f2_grid = tk.Frame(self.frame_filter_burst)
        self.frame_vs_plot_f2_grid.grid(row=2, column=6, columnspan=6)

        self.frame_vs_plot_f2 = tk.Frame(self.frame_vs_plot_f2_grid)
        self.frame_vs_plot_f2.pack(side="top", fill="both", expand=True)

        self.filter_2_graph = Graph_filter(self.frame_vs_plot_f2, self.controller.view, self.controller, self.filter_2_low_sv, self.filter_2_high_sv, figsize=(7,2), dpi=75)



        ttk.Button(self.frame_filter_burst, text="Filter", command=self.filter_burst).grid(row=3, column=0, columnspan=6,  padx=pad_filter)
        ttk.Button(self.frame_filter_burst, text="CLEAR Filter", command=self.clear_filter).grid(row=3, column=6, columnspan=6,
                                                                                           padx=pad_filter)


        #measurement type
        self.frame_measurement_type = tk.LabelFrame(self.frame_filter, text="Measurement type",
                                                    borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_measurement_type.pack(side="top", fill="both", expand=True)

        self.cb_meas_type_sv = tk.StringVar()
        self.cb = ttk.Combobox(self.frame_measurement_type, width=25, justify=tk.CENTER, textvariable=self.cb_meas_type_sv, values='')
        self.cb['values'] = ('lifetime, PTOFS')
        self.cb.set('lifetime')
        self.cb.bind('<<ComboboxSelected>>', self.change_meas_type)
        self.cb.grid(row=0, column=0)

        ttk.Button(self.frame_measurement_type, text="Calculate", command=self.calculate_measurement).grid(row=0, column=1, padx=5)

        self.check_is_ini_guess_iv = tk.IntVar(value=1)
        ttk.Checkbutton(self.frame_measurement_type, text="Ini guess ?", variable=self.check_is_ini_guess_iv).grid(row=1, column=0)

        ttk.Button(self.frame_measurement_type, text="Fit", command=self.fit_measurement).grid(row=1, column=1, padx=5)
        #TODO guess ? additionnal parameters
        #TODO scatter graph with fit results. (3D, surface, ...)


        self.frame_plot_notebook = tk.Frame(self.notebook)
        self.frame_plot_notebook.pack(side="top", fill="both", expand=True)
        self.notebook.add(self.frame_plot_notebook, text='Burst Analysis')

        self.frame_plot = tk.LabelFrame(self.frame_plot_notebook, text="Plot",
                                        borderwidth=self.appearence_param.frameLabelBorderWidth)

        self.frame_plot.pack(side="top", fill="both", expand=True)

        self.combo_box_plot_x_sv = tk.StringVar()
        self.cb_plot_x = ttk.Combobox(self.frame_plot, width=15, justify=tk.CENTER, textvariable=self.combo_box_plot_x_sv,
                          values='', state='readonly')
        self.cb_plot_x.bind('<<ComboboxSelected>>', self.change_x_plot)
        #TODO method to set the combo box
        self.cb_plot_x['values'] = ["duration", "nb_photon", "CPS", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
        self.combo_box_plot_x_sv.set("duration")
        self.cb_plot_x.grid(row=0, column=0, padx=pad_filter)

        ttk.Label(self.frame_plot, text='vs').grid(row=0, column=1, padx=pad_filter)

        self.combo_box_plot_y_sv = tk.StringVar()
        self.cb_plot_y = ttk.Combobox(self.frame_plot, width=15, justify=tk.CENTER, textvariable=self.combo_box_plot_y_sv,
                          values='', state='readonly')
        self.cb_plot_y.bind('<<ComboboxSelected>>', self.change_y_plot)
        #TODO method to set the combo box
        self.cb_plot_y['values'] = ["duration", "nb_photon", "CPS", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
        self.combo_box_plot_y_sv.set("duration")
        self.cb_plot_y.grid(row=0, column=2, padx=pad_filter)

        ttk.Button(self.frame_plot, text="Plot", command=self.plot).grid(row=0, column=3)

        # master_frame, view, controller, figsize, dpi, createSpan = True, createCursors = False):
        self.frame_vs_plot = tk.Frame(self.frame_plot)
        self.frame_vs_plot.grid(row=1, column=0, columnspan=4)

        self.figure_plotvs = plt.Figure(figsize=(7,3), dpi=180)
        self.ax_plotvs = self.figure_plotvs.add_subplot(111)

        self.figure_plotvs.set_tight_layout(True)


        self.canvas_plotvs = FigureCanvasTkAgg(self.figure_plotvs, master=self.frame_vs_plot)
        self.toolbar_plotvs = NavigationToolbar2Tk(self.canvas_plotvs, self.frame_vs_plot)
        self.canvas_plotvs.get_tk_widget().pack(side='top', fill='both', expand=1)

        #FIXME
        # NB it is here because its callback need the existence of self.filter_1_graph
        # tk.Checkbutton(self.frame_tree_params, text='show Filtered as gray', variable=self.check_show_filtered_iv,
        #                command=self.update_ui(), onvalue=1, offvalue=0).grid(row=0, column=0, columnspan=2)


    def bin_signal(self):
        bin_in_ms = int(self.binsize_sv.get())
        bin_in_tick = (bin_in_ms /1000.0) / self.burst_measure.exp_param.mAcrotime_clickEquivalentIn_second
        # data = self.controller.current_exp.data
        # timestamps = data.channels[self.burst_measure.num_channel].photons['timestamps']
        self.PCH = self.burst_measure.bin(bin_in_tick)

        #Display PCH on the dedicated graph.
        self.pch_graph.plot(self.PCH)


    def toggle_not_f1(self):
        if self.is_not_f1:
            self.button_not_f1.config(font=('courier', 12, 'normal'), foreground='black', fg='gray50')
            self.is_not_f1 = False
        else:
            self.button_not_f1.config(font=('courier', 12, 'bold'), foreground='black', fg='gray0')
            self.is_not_f1 = True

    def toggle_not_f2(self):
        if self.is_not_f2:
            self.button_not_f2.config(font=('courier', 12, 'normal'), foreground='black', fg='gray50')
            self.is_not_f2 = False
        else:
            self.button_not_f2.config(font=('courier', 12, 'bold'), foreground='black', fg='gray0')
            self.is_not_f2 = True

    def change_model(self):
        pass

    def change_x_plot(self, event=None):
        pass

    def change_y_plot(self, event=None):
        pass

    def update_ui(self):
        self.insert_measurement_tree_view(self.burst_measure)
        self.filter_1_graph.replot()
        self.filter_2_graph.replot()
        self.plot()

    def clear_filter(self):
        self.burst_measure.clear_filter()
        self.update_ui()

    def get_selected_burst_from_treeview(self):
        id_selected_item = self.tree_view.focus()
        parent_iid = self.tree_view.parent(id_selected_item)
        selected_item = self.tree_view.item(id_selected_item)

        #Have we clicked on a burst measurement or a burst ?
        if parent_iid=="":
            # This is a measure
            return None
        else:
            # This is a burst
            # Burst are retreived via their id(). See insert method
            num_burst = 0
            for id_burst, iid_tk in self.burst_iid_dict.items():
                if iid_tk == id_selected_item:
                    for burst in self.burst_measure.bursts:
                        if id_burst == id(burst):
                            break
                        else:
                            num_burst += 1
            return self.burst_measure.bursts[num_burst]

    def toggle_filter_burst(self):
        burst = self.get_selected_burst_from_treeview()
        burst.is_filtered = not burst.is_filtered
        self.update_ui()

    def toggle_highlight_burst(self):
        burst = self.get_selected_burst_from_treeview()
        burst.is_highlighted = not burst.is_highlighted
        self.update_ui()


    def filter_burst(self):

        def convert_sv_float(sv):
            str_ = sv.get()
            try:
                f = float(str_)
                return f
            except ValueError:
                return 0
        low1 = convert_sv_float(self.filter_1_low_sv)
        high1 = convert_sv_float(self.filter_1_high_sv)
        type1 = self.cb_value_filter_1_sv.get()

        bool_op = self.cb_value_filter_bool_op_sv.get()

        type2 = self.cb_value_filter_2_sv.get()
        low2 = convert_sv_float(self.filter_2_low_sv)
        high2 = convert_sv_float(self.filter_2_high_sv)
        self.burst_measure.filter(low1, high1, type1, self.is_not_f1, bool_op, low2, high2, type2, self.is_not_f2)

        self.update_ui()

    def plot(self):
        self.ax_plotvs.clear()

        x_val = self.combo_box_plot_x_sv.get()
        y_val = self.combo_box_plot_y_sv.get()

        if x_val == y_val:
            if x_val == "duration":
                self.ax_plotvs.set_xlabel("duration")
                self.ax_plotvs.plot(self.burst_measure.bin_edges_bursts_length[0:-1], self.burst_measure.bursts_length_histogram)
            elif x_val == "nb_photon":
                self.ax_plotvs.set_xlabel("Nb photon")
                self.ax_plotvs.plot(self.burst_measure.bin_edges_bursts_intensity[0:-1],
                                    self.burst_measure.bursts_intensity_histogram)
            elif x_val == "CPS":
                self.ax_plotvs.set_xlabel("CPS")
                self.ax_plotvs.plot(self.burst_measure.bin_edges_bursts_CPS[0:-1], self.burst_measure.bursts_CPS_histogram)
            self.ax_plotvs.set_ylabel("Occurence")
            self.figure_plotvs.canvas.draw()
        else:
            x = np.zeros(len(self.burst_measure.bursts))
            y = np.zeros(len(self.burst_measure.bursts))

            if x_val == "duration":
                self.ax_plotvs.set_xlabel("duration")
                for i, burst in enumerate(self.burst_measure.bursts):
                    x[i] = burst.duration_tick
            elif x_val == "nb_photon":
                self.ax_plotvs.set_xlabel("Nb photon")
                for i, burst in enumerate(self.burst_measure.bursts):
                    x[i] = burst.nb_photon
            elif x_val == "CPS":
                self.ax_plotvs.set_xlabel("CPS")
                for i, burst in enumerate(self.burst_measure.bursts):
                    x[i] = burst.CPS

            if y_val == "duration":
                self.ax_plotvs.set_ylabel("duration")
                for i, burst in enumerate(self.burst_measure.bursts):
                    y[i] = burst.duration_tick
            elif y_val == "nb_photon":
                self.ax_plotvs.set_ylabel("Nb photon")
                for i, burst in enumerate(self.burst_measure.bursts):
                    y[i] = burst.nb_photon
            elif y_val == "CPS":
                self.ax_plotvs.set_ylabel("CPS")
                for i, burst in enumerate(self.burst_measure.bursts):
                    y[i] = burst.CPS


            self.ax_plotvs.scatter(x, y)
            self.figure_plotvs.canvas.draw()



    def auto_threshold(self):
        """
        Based on transient Event Detection over Background noise.

        The
        TODO should be elsewhere in the core part of the software
        :return:
        """
        # def poisson(x, mu):
        #     return np.exp(-mu) / factorial(x) * np.power(mu, x)

        false_negative_ratio = float(self.false_negative_sv.get()) / 100.0
        mu = self.PCH.time_axis[np.argmax(self.PCH.data)]
        chi = 0
        while 1 - poisson.cdf(chi, mu) > false_negative_ratio:
            chi += 1

        self.pch_graph.threshold = chi
        self.threshold_flank_sv.set(str(chi))

        # NB this value is hardcoded !
        false_negative_ratio = false_negative_ratio/1000.0

        mu = self.PCH.time_axis[np.argmax(self.PCH.data)]
        chi = 0
        while 1 - poisson.cdf(chi, mu) > false_negative_ratio:
            chi += 1

        self.pch_graph.threshold_burst = chi
        self.threshold_burst_sv.set(str(chi))

        self.pch_graph.plot(self.PCH)

    def launch_detection(self):
        threshold_burst = int(self.threshold_burst_sv.get())
        threshold_flank = int(self.threshold_flank_sv.get())
        min_succesive_bin = int(self.min_succesive_bin_sv.get())
        max_succesive_noise_bin = int(self.max_succesive_noise_bin_sv.get())
        min_nb_photon = int(self.min_nb_photon_sv.get())

        self.burst_measure.do_threshold(threshold_burst, threshold_flank, min_succesive_bin, max_succesive_noise_bin, min_nb_photon)

        # Display burst infos
        self.nb_of_burst_sv.set(str(self.burst_measure.get_nb_of_burst()))
        self.nb_of_short_burst_sv.set(str(self.burst_measure.nb_too_short_burst))

        # Plot

        #TODO noise

        # self.noise_CPS_graph.plot()
        #
        # self.noise_duration_graph.plot()

        self.burst_int_graph.plot((self.burst_measure.bin_edges_bursts_intensity, self.burst_measure.bursts_intensity_histogram), "burst_int")

        self.burst_duration_graph.plot(
            (self.burst_measure.bin_edges_bursts_length, self.burst_measure.bursts_length_histogram), "burst_length")
        self.burst_CPS_graph.plot(
            (self.burst_measure.bursts_length, self.burst_measure.bursts_CPS), "burst_CPS")


    def next_burst(self):
        # plcer des barres/lignes sur les stats
        num_burst = int(self.num_burst_sv.get()) + 1
        burst = self.burst_measure.get_burst(num_burst)

        self.nb_photon_burst_sv.set(str(burst.nb_photon))
        self.length_burst_sv.set(str(burst.duration_tick * self.burst_measure.exp_param.mAcrotime_clickEquivalentIn_second * 1E6))
        self.cps_burst_sv.set(str(int(burst.CPS)))

        self.controller.display_burst(burst, self.burst_measure)
        self.num_burst_sv.set(str(num_burst))

    def previous_burst(self):
        num_burst = int(self.num_burst_sv.get()) - 1
        burst = self.burst_measure.get_burst(num_burst)
        self.controller.display_burst(burst, self.burst_measure)
        self.num_burst_sv.set(str(num_burst))

    def validate(self):
        self.controller.validate_burst_selection(self.burst_measure)
        self.insert_measurement_tree_view(self.burst_measure)

    def change_filter1_type(self, event=None):
        type_ = self.cb_value_filter_1_sv.get()
        self.filter_1_graph.plot(self.burst_measure, type_)


    def change_filter2_type(self, event=None):
        type_ = self.cb_value_filter_2_sv.get()
        self.filter_1_graph.plot(self.burst_measure, type_)


    def clear_treeview(self):
        self.tree_view.delete(*self.tree_view.get_children())



    def insert_measurement_tree_view(self, burst_measurement):
        #FIXME j'ai melangé deux methodes pour inserer UN burst et tous les bursts ?
        self.clear_treeview()


        # iid = self.tree_view.insert(parent="", index='end', text=exp.file_name)
        # ("name", "num burst", "tick start", "tick stop", "nb photon", "CPS", "channel", "m. type", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")
        iid_measurement = self.tree_view.insert(parent="", index='end', values=(
            burst_measurement.name, "", "", "", "", "", burst_measurement.num_channel, "", "", "", "", "", "", "", "", ""))
        if burst_measurement.name not in self.measurement_iid_dict:
            self.measurement_iid_dict[burst_measurement.name] = iid_measurement
        self.tree_view.focus(iid_measurement)
        num = 0
        for burst in burst_measurement.bursts:
            tags_ = []
            if burst.is_filtered:
                tags_.append("filtered")
                if not self.check_show_filtered_iv.get():
                    continue
                #TODO depend on checkbox
                # continue
            if burst.is_highlighted:
                tags_.append("highlighted")

            if burst.measurement is None:
                type = "None"
                p1,p2,p3,p4,p5,p6,p7,p8 = "","","","","","","",""

            iid = self.tree_view.insert(parent=iid_measurement, index='end',
                                        values=(
                                            "", num, burst.tick_start, burst.duration_tick, burst.nb_photon, int(burst.CPS),
                                            burst_measurement.num_channel,
                                            type, p1,  p2, p3, p4, p5, p6, p7, p8,), tags=tags_)
            self.tree_view.item(iid_measurement, open=True)
            # we use id() as a single identificator of the burst
            if id(burst) not in self.burst_iid_dict:
                self.burst_iid_dict[id(burst)] = iid
            num += 1



    def treeview_measurement_select(self, event):
        self.current_burst = self.controller.current_burst = self.get_selected_burst_from_treeview()
        self.controller.display_burst(self.current_burst, self.burst_measure)

        # item_num_burst = selected_item["values"][1]
        # item_name_mes = selected_item["values"][2]
        # item_name_burst = selected_item["values"][1]

        # if item_name_exp in self.controller.model.experiments:
        #     # this is an experiment
        #     exp = self.controller.set_current_exp(item_name_exp)
        #     self.controller.update_navigation()
        # elif item_name_burst != "":
        #     parent_iid = self.tree_view.parent(id_selected_item)
        #     if item_name_mes in self.controller.current_exp.measurements:
        #         measurement = self.controller.set_current_measurement(item_name_mes)
        #         num_burst = int(item_name_burst[2:])
        #         self.controller.display_burst(measurement.bursts[num_burst], measurement)
        #
        # elif item_name_mes in self.controller.current_exp.measurements:
        #     # this is a measurement
        #     measurement = self.controller.set_current_measurement(item_name_mes)
        #     self.controller.view.archi.analyze_area.display_measurement(measurement)
        #     self.controller.display_measurement(measurement.name)


    def change_meas_type(self, event=None):
        type = self.cb_meas_type_sv.get()

        # Il faut des paramètres, des modeles, des xlim, une iRf, etc...
        self.burst_measure.perform_measurements(type)

        self.display_measurement(self.burst_measure, type)

    def display_measurement(self, burst_measurement, type):
        return
        for child in self.frame_measurement_params.winfo_children():
            child.destroy()

        if type == "chronogram":
            pass
        elif type == "lifetime":
            pass
        #
        #     lifetime_measurement = None
        #
        #     self.analyze_gui = lifeTimeAnalyze_gui(self.frame_measurement_params, self.controller, self.appearence_param, lifetime_measurement, is_burst_analysis=True)
        #     self.analyze_gui.populate()
        #
        #     if lifetime_measurement.IR_raw is not None:
        #         self.analyze_gui.isDraw_IR.set(lifetime_measurement.use_IR)
        #         self.analyze_gui.ir_name_sv.set(lifetime_measurement.IR_name)
        #         self.analyze_gui.ir_start_sv.set(str(lifetime_measurement.IR_start))
        #         self.analyze_gui.ir_end_sv.set(str(lifetime_measurement.IR_end))
        #         self.analyze_gui.shiftIR_amount_sv.set(str(lifetime_measurement.IR_shift))
        #         self.analyze_gui.bckg_IR_sv.set(str(lifetime_measurement.IR_bckg))
        #
        #     self.gui_for_fit_operation = self.analyze_gui.gui_for_fit_operation
        #
        # self.frame_measurement_params.pack(side="top", fill="both", expand=True)

    def calculate_measurement(self):
        type = self.cb_meas_type_sv.get()

        # Il faut des paramètres, des modeles, des xlim, une iRf, etc...
        self.burst_measure.perform_measurements(type)

        self.display_measurement(self.burst_measure, type)
        # Update the treeview


    def fit_measurement(self):
        gui_for_fit_operation = self.controller.view.archi.analyze_area.gui_for_fit_operation
        fit_params = gui_for_fit_operation.get_fit_params()
        # self.controller.launch_burst_measurement(self.burst_measure, type, model_name, fit_params, xlim_min_fit,
        #                                          xlim_max_fit)
        is_ini_guess = self.check_is_ini_guess_iv.get()
        self.burst_measure.perform_fit_of_measurement(fit_params, is_ini_guess)

        # Update the treeview

    def on_double_click_treeview(self, event):
        region = self.tree_view.identify("region", event.x, event.y)
        if region == "heading":
            # Returns the data column identifier of the cell at position x. The tree column has ID #0.
            column_id = self.tree_view.identify_column(event.x)
            print(column_id)

        #TODO sort column https://stackoverflow.com/questions/1966929/tk-treeview-column-sort

from .graph.interactiveGraphs import InteractiveGraph
from matplotlib.widgets import Cursor

class Graph_PCH_burst(InteractiveGraph):
    """
    doc todo
    """

    def __init__(self, master_frame, view, controller, burst_GUI, figsize, dpi):
        super().__init__(master_frame, view, controller, figsize, dpi)
        #self.ax.axis('off')
        self.figure.tight_layout()
        self.burst_GUI = burst_GUI

        self.threshold = None
        self.threshold_burst = None
        self.pch = None

        self.createCallBacks()
        self.createWidgets()

    def plot(self, PCH):
        self.pch = PCH

        if self.ax is None:
            self.mainAx = self.figure.add_subplot(111)
            self.subplot3D = None
        self.ax.clear()

        # # reduce nb of point to 1000 (approximative size in pixel
        # skipsize = int(PCH.nbOfBin / 1000)
        # idx = np.arange(0, len(PCH.data), skipsize)


        # Compare to the equivalent poisson distribution
        eq_poisson = poisson.pmf(PCH.time_axis, mu=np.max(PCH.data))
        eq_poisson *= np.max(PCH.data)/np.max(eq_poisson)



        if np.max(PCH.data) > 10*np.mean(PCH.data):
            self.ax.loglog(PCH.time_axis, PCH.data)
            # self.ax.loglog(PCH.time_axis, eq_poisson, "k--", alpha=0.5)
            self.ax.set_xlim(1, np.max(PCH.time_axis))
        else:
            self.ax.semilogy(PCH.time_axis, PCH.data)
            # self.ax.semilogy(PCH.time_axis, eq_poisson, "k--", alpha=0.5)


        if self.threshold is not None:
            self.ax.vlines(self.threshold, 0, PCH.data.max(), linewidth=4)
        if self.threshold_burst is not None:
            self.ax.vlines(self.threshold_burst, 0, PCH.data.max(), linewidth=4)

        self.figure.canvas.draw()

    def button_press_event(self, event):
        if event.button == 1:
            self.threshold = int(event.xdata)
            self.burst_GUI.threshold_flank_sv.set(str(self.threshold))
            self.plot(self.pch)



    def onSpanMove(self, xmin, xmax):
        pass

    def onSpanSelect(self, xmin, xmax):
        pass

    def scrollEvent(self, event):
        pass

    def createWidgets(self):
        # super().createWidgets()
        self.cursor_h = Cursor(self.ax, useblit=True, color='red', horizOn=False, vertOn=True, linewidth=3)

        # self.cursor_h.set_active(False)
        # self.cursor_h.drawon = True
        # drawon
        # eventson
        # self.setOnOffCursors(True)


class Graph_stat(InteractiveGraph):
    """
    doc todo
    """

    def __init__(self, master_frame, view, controller, burst_GUI, figsize, dpi):
        super().__init__(master_frame, view, controller, figsize, dpi)
        #self.ax.axis('off')
        self.figure.tight_layout()
        self.burst_GUI = burst_GUI

        self.createCallBacks()
        self.createWidgets()

    def plot(self, stat, type):
        self.stat = stat

        if self.ax is None:
            self.mainAx = self.figure.add_subplot(111)
            self.subplot3D = None
        self.ax.clear()

        if type == "burst_CPS":
            self.ax.scatter(stat[0], stat[1])
            self.ax.set_xlabel("length")
            self.ax.set_ylabel("CPS")
        elif type == "burst_length":
            self.ax.plot(stat[0][0:-1], stat[1])
            self.ax.set_xlabel("length (µs)")
            self.ax.set_ylabel("Occurence")

        elif type == "burst_int":
            self.ax.plot(stat[0][0:-1], stat[1])
            self.ax.set_xlabel("Nb of Photon")
            self.ax.set_ylabel("Occurence")

        self.figure.canvas.draw()

    def button_press_event(self, event):
        if event.button == 1:
            # self.threshold = int(event.xdata)
            # self.burst_GUI.threshold_sv.set(str(self.threshold))
            # self.plot(self.pch)
            pass

    def createWidgets(self):
        # super().createWidgets()
        self.cursor_h = Cursor(self.ax, useblit=True, color='red', horizOn=False, vertOn=True, linewidth=3)



        # self.cursor_h.set_active(False)
        # self.cursor_h.drawon = True
        # drawon
        # eventson
        # self.setOnOffCursors(True)


class Graph_filter(InteractiveGraph):
    """
    doc todo
    """

    def __init__(self, master_frame, view, controller, low_sv, high_sv, figsize, dpi):
        super().__init__(master_frame, view, controller, figsize, dpi)
        #self.ax.axis('off')
        self.figure.tight_layout()
        self.low_sv = low_sv
        self.high_sv = high_sv

        self.burst_measure = None
        self.type_ = None

        self.span_min = 0
        self.span_max = 0

        self.createCallBacks()
        self.createWidgets()

    def plot(self, burst_measure, type_):
        self.burst_measure = burst_measure
        self.type_ = type_

        if self.ax is None:
            self.ax = self.figure.add_subplot(111)

        self.ax.clear()

        if type_ == "duration":
            self.ax.set_xlabel("duration")
            x = self.burst_measure.bin_edges_bursts_length[0:-1]
            y = self.burst_measure.bursts_length_histogram
        elif type_ == "nb_photon":
            self.ax.set_xlabel("Nb photon")
            x = self.burst_measure.bin_edges_bursts_intensity[0:-1]
            y = self.burst_measure.bursts_intensity_histogram
        elif type_ == "CPS":
            self.ax.set_xlabel("CPS")
            x = self.burst_measure.bin_edges_bursts_CPS[0:-1]
            y = self.burst_measure.bursts_CPS_histogram
        self.ax.set_ylabel("Occurence")
        max_data = np.max(y)
        self.ax.plot(x, y)
        #TODO other measurement and parameters



        self.ax.add_patch(
            patches.Rectangle(
                (self.span_min, 0),  # (x,y)
                self.span_max-self.span_min,  # width
                max_data,  # height
                alpha=0.2
            )
        )

        self.figure.canvas.draw()

    def replot(self):
        if self.burst_measure is not None:
            self.plot(self.burst_measure, self.type_)

    def button_press_event(self, event):
        if event.button == 1:
            # self.threshold = int(event.xdata)
            # self.burst_GUI.threshold_sv.set(str(self.threshold))
            # self.plot(self.pch)
            pass

    # def createWidgets(self):
    #     # super().createWidgets()
    #     self.cursor_v1 = Cursor(self.ax, useblit=True, color='red', horizOn=False, vertOn=True, linewidth=3)
    #     self.cursor_v2 = Cursor(self.ax, useblit=True, color='blue', horizOn=False, vertOn=True, linewidth=3)
    #
    #     # self.cursor_h.set_active(False)
    #     # self.cursor_h.drawon = True
    #     # drawon
    #     # eventson
    #     # self.setOnOffCursors(True)

    def onSpanSelect(self, xmin, xmax):
        self.span_min = xmin
        self.span_max = xmax
        self.low_sv.set(str(xmin))
        self.high_sv.set(str(xmax))
        self.replot()






    def onSpanMove(self, xmin, xmax):
        pass




