

import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as tkst
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from GUI.graph.Graph_Results import Graph_Results

from .analyze_Lifetime import lifeTimeAnalyze_gui
from .graph.interactiveGraphs import InteractiveGraph
import numpy as np
from scipy.special import factorial
from scipy.stats import poisson
import matplotlib.patches as patches

class LifetimeBench_gui():
	def __init__(self, master_frame, controller, appearence_param, lifetime_bench=None):
		self.master_frame = master_frame
		self.controller = controller
		self.appearence_param = appearence_param
		self.lifetime_bench = lifetime_bench
		self.modelNames = ['One Decay Tail A1', 'One Decay IRF', 'One Decay Tail', 'Two Decays IRF', 'Two Decays IRF A1 A2', 'Two Decays Tail', 'Two Decays Tail A1 A2', 'IRF Becker', "MonoExp for IRF", "Three Decays Tail A1 A2 A3"]
		self.bench_iid_dict = {}

		self.nb_max_param_bench = 12

		self.list_label_sv_param_fit = []
		self.nb_max_param_fit = 8

		self.populate()


	def populate(self):
		self.top_level = tk.Toplevel(self.master_frame)
		self.top_level.title("Lifetime Fitting Benchmark")

		self.notebook = ttk.Notebook(self.top_level)
		self.notebook.pack(expand=True, fill="both")

		self.frame_input_param = tk.Frame(self.notebook)
		self.frame_input_param.pack(side="top", fill="both", expand=True)
		self.notebook.add(self.frame_input_param, text='Input Parameter')

		self.cmd_fit_frame = ttk.Frame(self.frame_input_param)
		self.cmd_fit_frame.pack(side="top", fill="both", expand=True)


		label = ttk.Label(self.cmd_fit_frame, text='Fit Model :')
		label.grid(row=0, column=0)

		self.cb_model_sv = tk.StringVar()
		cb = ttk.Combobox(self.cmd_fit_frame, width=15, justify=tk.CENTER, textvariable=self.cb_model_sv,
						  values='', state='readonly')
		cb.bind('<<ComboboxSelected>>', self.change_model)
		cb['values'] = self.modelNames
		self.cb_model_sv.set("One Decay Tail A1")
		cb.set(self.modelNames[0])
		cb.grid(row=0, column=1, columnspan=3)

		# Formula
		self.formulaFrame = tk.Frame(master=self.cmd_fit_frame)
		self.formulaFrame.grid(row=3, column=0, columnspan=4)

		self.figTex = plt.Figure(figsize=(13, 2), dpi=28, frameon=False)
		self.axTex = self.figTex.add_axes([0, 0, 1, 1])

		self.axTex.axis('off')

		self.canvasTk = FigureCanvasTkAgg(self.figTex, master=self.formulaFrame)
		self.canvasTk.get_tk_widget().pack(side='top', fill='both', expand=1)

		# Quantity to minimize
		ttk.Label(self.cmd_fit_frame, text='Qty to minimize').grid(row=5, column=0)
		self.cb_minqty_to_min_sv = tk.StringVar()
		cb = ttk.Combobox(self.cmd_fit_frame, width=15, justify=tk.CENTER, textvariable=self.cb_minqty_to_min_sv,
						  values='', state='readonly')
		cb.bind('<<ComboboxSelected>>', self.change_minqty_to_min)
		cb['values'] = ["auto", "chi2", "max. likelyhood (MLE)"]
		self.cb_minqty_to_min_sv.set("auto")
		cb.set("auto")
		cb.grid(row=5, column=1)

		# Methods
		ttk.Label(self.cmd_fit_frame, text='Method 1').grid(row=6, column=0)
		self.cb_method1_sv = tk.StringVar()
		cb = ttk.Combobox(self.cmd_fit_frame, width=15, justify=tk.CENTER, textvariable=self.cb_method1_sv,
						  values='', state='readonly')
		cb.bind('<<ComboboxSelected>>', self.change_method1)
		cb['values'] = ["least_squares", "leastsq", "differential_evolution", "brute", "basinhopping", "ampgo", "nelder", "lbfgsb", "powell", "cg", "newton", "cobyla", "bfgs", "tnc", "trust-ncg", "trust-exact", "trust-krylov", "trust-constr", "dogleg", "slsqp", "emcee", "shgo", "dual_annealing"]
		self.cb_method1_sv.set("least_squares")
		cb.set("least_squares")
		cb.grid(row=6, column=1)

		ttk.Label(self.cmd_fit_frame, text='Method 2').grid(row=6, column=2)
		self.cb_method2_sv = tk.StringVar()
		cb = ttk.Combobox(self.cmd_fit_frame, width=15, justify=tk.CENTER, textvariable=self.cb_method2_sv,
						  values='', state='readonly')
		cb.bind('<<ComboboxSelected>>', self.change_method2)
		cb['values'] = ["None", "leastsq", "least_squares", "differential_evolution", "brute", "basinhopping", "ampgo", "nelder", "lbfgsb", "powell", "cg", "newton", "cobyla", "bfgs", "tnc", "trust-ncg", "trust-exact", "trust-krylov", "trust-constr", "dogleg", "slsqp", "emcee", "shgo", "dual_annealing"]
		self.cb_method2_sv.set("None")
		cb.set("None")
		cb.grid(row=6, column=3)

		ttk.Label(self.cmd_fit_frame, text='Nb time bins').grid(row=7, column=0)
		self.nb_time_bins_sv = tk.StringVar(value=str(4096))
		ttk.Entry(self.cmd_fit_frame, textvariable=self.nb_time_bins_sv, justify=tk.CENTER, width=12).grid(row=7, column=1)

		ttk.Label(self.cmd_fit_frame, text='time bin duration (ps)').grid(row=7, column=2)
		self.duration_time_bins_sv = tk.StringVar(value=str(6.10))
		ttk.Entry(self.cmd_fit_frame, textvariable=self.duration_time_bins_sv, justify=tk.CENTER, width=12).grid(row=7, column=3)


		self.check_is_ini_guess_iv = tk.IntVar(value=1)
		tk.Checkbutton(self.cmd_fit_frame, text='Ini guess ?', variable=self.check_is_ini_guess_iv, onvalue=1, offvalue=0).grid(row=8, column=0, columnspan=2)

		self.check_is_fit_iv = tk.IntVar(value=1)
		tk.Checkbutton(self.cmd_fit_frame, text='Fit ?', variable=self.check_is_fit_iv, onvalue=1, offvalue=0).grid(row=8, column=2, columnspan=2)

		# fit boundaries
		ttk.Label(self.cmd_fit_frame, text='fit boundaries (x1, x2)').grid(row=9, column=0)

		self.idx_lim_for_fit_min_sv = tk.StringVar()
		ttk.Entry(self.cmd_fit_frame, textvariable=self.idx_lim_for_fit_min_sv, justify=tk.CENTER, width=12).grid(row=9,
																											  column=1)

		self.idx_lim_for_fit_max_sv = tk.StringVar()
		ttk.Entry(self.cmd_fit_frame, textvariable=self.idx_lim_for_fit_max_sv, justify=tk.CENTER, width=12).grid(row=9,
																											  column=2)

		self.cmd_fit_frame.pack(side="top", fill="both", expand=True)

		# Parameters
		self.param_frame = ttk.Frame(self.frame_input_param)

		# column header
		ttk.Label(self.param_frame, text='min').grid(row=1, column=0)
		ttk.Label(self.param_frame, text='max').grid(row=2, column=0)

		# Labels for parameters
		# self.list_label_sv_param = [tk.StringVar()] * self.nb_max_param_bench
		self.list_label_sv_param = [tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()]
		for i in range(self.nb_max_param_bench):
			ttk.Label(self.param_frame, text="", textvariable=self.list_label_sv_param[i]).grid(row=0, column=1 + i)

		self.entry_text_size = 10



		self.param_frame.pack(side="top", fill="both", expand=True)



		# Fit parameters (limits, fix value and so on).

		# Parameters
		self.param_fit_frame = ttk.Frame(self.frame_input_param)

		# column header
		ttk.Label(self.param_fit_frame, text='').grid(row=0, column=0)
		ttk.Label(self.param_fit_frame, text='value').grid(row=0, column=1)
		ttk.Label(self.param_fit_frame, text='').grid(row=0, column=2)
		ttk.Label(self.param_fit_frame, text='').grid(row=0, column=3)
		ttk.Label(self.param_fit_frame, text='min').grid(row=0, column=4)
		ttk.Label(self.param_fit_frame, text='max').grid(row=0, column=5)
		ttk.Label(self.param_fit_frame, text='b step').grid(row=0, column=6)
		ttk.Label(self.param_fit_frame, text='vary').grid(row=0, column=7)

		# Labels for parameters
		self.list_label_sv_param_fit = [tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(),
									tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()]
		for i in range(self.nb_max_param_fit):
			ttk.Label(self.param_fit_frame, text="", textvariable=self.list_label_sv_param_fit[i]).grid(row=1 + i, column=0)

		self.entry_text_size = 10

		# model_name = self.cb_model_sv.get()
		# self.measurement.set_model(model_name)
		# self.create_gui_from_measurement_params()

		self.param_fit_frame.pack(side="top", fill="both", expand=True)

		self.launch_bench_frame = ttk.Frame(self.frame_input_param)

		ttk.Label(self.launch_bench_frame, text='Nb decays').grid(row=0, column=0)
		self.nb_decays_sv = tk.StringVar(value=str(10))
		ttk.Entry(self.launch_bench_frame, textvariable=self.nb_decays_sv, justify=tk.CENTER, width=12).grid(row=0, column=1)

		ttk.Button(self.launch_bench_frame, text="Launch Benchmark", command=self.launch_bench).grid(row=0, column=2)
		self.launch_bench_frame.pack(side="top", fill="both", expand=True)

		self.change_model(None)

		#TODO :
		# - un treeview avec les bench avec les parametres (chi2, erreur param fit) où on eut voir le fit de chacun en cliquant dessus
		# - Filtre selon certain critère
		# - un score général
		# - Plot

		# FRAME ANALYSIS
		self.frame_filter = tk.Frame(self.notebook)
		self.frame_filter.pack(side="top", fill="both", expand=True)
		self.notebook.add(self.frame_filter, text='Bench Analysis')

		#burst list
		self.frame_bench_list = tk.LabelFrame(self.frame_filter, text="Bench List",
											  borderwidth=self.appearence_param.frameLabelBorderWidth)
		self.frame_bench_list.pack(side="top", fill="both", expand=True)

		#https://riptutorial.com/tkinter/example/31885/customize-a-treeview
		self.tree_view = ttk.Treeview(self.frame_bench_list)
		self.tree_view["columns"] = ("num", "r. chi2", "nb photon", "noise", "irf length", "score", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10")
		# remove first empty column with the identifier
		# self.tree_view['show'] = 'headings'
		# tree.column("#0", width=270, minwidth=270, stretch=tk.NO) tree.column("one", width=150, minwidth=150, stretch=tk.NO) tree.column("two", width=400, minwidth=200) tree.column("three", width=80, minwidth=50, stretch=tk.NO)

		columns_text = ["num", "r. chi2", "nb photon", "noise", "irf length", "score", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p4"]
		self.tree_view.column("#0", width=25, stretch=tk.NO)
		self.tree_view.column(columns_text[0], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[1], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[2], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[3], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[4], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[5], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[6], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[7], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[8], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[9], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[10], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[11], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[12], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[13], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[14], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[15], width=50, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column("p11", width=50, stretch=tk.YES, anchor=tk.CENTER)



		self.tree_view.heading("num", text="num")
		for col in columns_text:
			self.tree_view.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(self.tree_view, _col, False))


		#FIXME only change text color to light gray
		self.tree_view.tag_configure('filtered', foreground='gray50')
		self.tree_view.tag_configure('filtered', background='gray20')

		self.tree_view.tag_configure('highlighted', background='gray90')
		self.tree_view.tag_configure('highlighted', foreground='gold3')

		ysb = ttk.Scrollbar(self.frame_bench_list, orient='vertical', command=self.tree_view.yview)
		self.tree_view.grid(row=0, column=0, sticky='nsew')
		ysb.grid(row=0, column=1, sticky='ns')
		self.tree_view.configure(yscroll=ysb.set)

		self.tree_view.bind('<<TreeviewSelect>>', self.treeview_measurement_select)
		self.tree_view.bind("<Double-1>", self.on_double_click_treeview)

		self.frame_tree_params = tk.Frame(self.frame_bench_list)
		self.frame_tree_params.grid(row=0, column=2, sticky='nsew')

		self.check_show_filtered_iv = tk.IntVar()
		tk.Checkbutton(self.frame_tree_params, text='show Filtered as gray', variable=self.check_show_filtered_iv, command=self.update_ui(), onvalue=1, offvalue=0).grid(row=0, column=0, columnspan=2)

		ttk.Label(self.frame_tree_params, text='Nb filtered Bench').grid(row=1, column=0)
		self.nb_filtered_burst_sv = tk.StringVar()
		e = ttk.Entry(self.frame_tree_params, textvariable=self.nb_filtered_burst_sv, justify=tk.CENTER, width=12)
		e.config(state=tk.DISABLED)
		e.grid(row=1, column=1)
		ttk.Label(self.frame_tree_params, text='/').grid(row=1, column=2)
		self.nb_total_burst_sv = tk.StringVar()
		e = ttk.Entry(self.frame_tree_params, textvariable=self.nb_total_burst_sv, justify=tk.CENTER, width=12)
		e.config(state=tk.DISABLED)
		e.grid(row=1, column=3)

		ttk.Button(self.frame_tree_params, text="Toggle Filtered", command=self.toggle_filter_bench).grid(row=2, column=0)
		ttk.Button(self.frame_tree_params, text="Toggle Highlighted", command=self.toggle_highlight_bench).grid(row=3,
																												column=0)

		# Score statistics
		self.frame_score_statistics = tk.Frame(self.frame_tree_params)
		ttk.Label(self.frame_score_statistics, text='mean').grid(row=0, column=0)
		ttk.Label(self.frame_score_statistics, text='median').grid(row=0, column=1)
		ttk.Label(self.frame_score_statistics, text='std').grid(row=0, column=2)
		self.score_mean_sv = tk.StringVar()
		e = ttk.Entry(self.frame_score_statistics, textvariable=self.score_mean_sv, justify=tk.CENTER, width=12)
		e.grid(row=1, column=0)
		self.score_median_sv = tk.StringVar()
		e = ttk.Entry(self.frame_score_statistics, textvariable=self.score_median_sv, justify=tk.CENTER, width=12)
		e.grid(row=1, column=1)
		self.score_std_sv = tk.StringVar()
		e = ttk.Entry(self.frame_score_statistics, textvariable=self.score_std_sv, justify=tk.CENTER, width=12)
		e.grid(row=1, column=2)

		self.frame_graph_score_stat = tk.Frame(self.frame_score_statistics)
		self.frame_graph_score_stat.grid(row=3, column=0, columnspan=3)

		self.figure_score_stat = plt.Figure(figsize=(3,2), dpi=100)
		self.ax_score_stat = self.figure_score_stat.add_subplot(111)

		self.canvas_score_stat = FigureCanvasTkAgg(self.figure_score_stat, master=self.frame_graph_score_stat)
		self.canvas_score_stat.get_tk_widget().pack(side='top', fill='both', expand=1)

		self.frame_score_statistics.grid(row=4, column=0, columnspan=2)

		#Bench fit graph
		self.frame_bench_graph = tk.LabelFrame(self.frame_filter, text="Bench Fit graph",
											   borderwidth=self.appearence_param.frameLabelBorderWidth)
		self.frame_bench_graph.pack(side="top", fill="both", expand=True)


		# Results from fit -- Text
		self.frame_left = tk.Frame(self.frame_bench_graph)
		self.frame_result_text = tk.LabelFrame(self.frame_left, text="Fit",
											   borderwidth=self.appearence_param.frameLabelBorderWidth)
		# self.frame_result_text.grid(row=0, column=0)
		self.frame_result_text.pack(side=tk.TOP, fill="both", expand=True)

		self.result_fit_text_area = tkst.ScrolledText(self.frame_result_text, wrap=tk.WORD, width=45, height=20)
		self.result_fit_text_area.pack(side=tk.LEFT, fill="both", expand=True)
		self.result_fit_text_area.insert(tk.INSERT, "Gimme Results !")

		self.frame_left.pack(side=tk.LEFT, fill="both", expand=True)

		self.frame_right = tk.Frame(self.frame_bench_graph)
		self.frame_analyze_graphs = tk.LabelFrame(self.frame_right, text="Graph",
												  borderwidth=self.appearence_param.frameLabelBorderWidth)

		self.frame_analyze_graphs.grid(row=0, column=1, sticky=tk.NSEW)
		self.graph_results = Graph_Results(self.frame_analyze_graphs, self.controller.view, self.controller,
										   figsize=(15, 6), dpi=100)

		self.frame_right.pack(side=tk.LEFT, fill="both", expand=True)



		"""
		# Filter bench
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
		"""






	def set_fit_formula(self, formula, fontsize=40):
		formula = "$" + formula + "$"

		self.axTex.clear()
		self.axTex.text(0, 0.2, formula, fontsize=fontsize)
		self.canvasTk.draw()

	def change_method1(self, event):
		pass

	def change_method2(self, event):
		pass

	def change_minqty_to_min(self, event):
		pass

	def change_model(self, event):
		model_name = self.cb_model_sv.get()

		self.lifetime_bench.set_model(model_name)
		self.create_gui_from_measurement_params()
		self.set_fit_formula(self.lifetime_bench.measurement_dummy.model.fit_formula)

	def launch_bench(self):
		params = self.get_bench_params()
		#TODO Thread because it is very very long and we need feedback
		self.controller.launch_bench(params)
		self.update_column_name_treeview()
		self.insert_benchmarks_treeview()
		self.set_score_statistics()

	def create_gui_from_measurement_params(self):
		self.gui_param_dict = {}
		self.gui_param_widget_dict = {}
		# self.changeModel(None)

		def create_value(key, label, num_column, default_min=0, default_max=10):
			self.gui_param_dict[key] = {}
			self.gui_param_dict[key]["min_gen"] = tk.StringVar(value=str(default_min))
			self.gui_param_dict[key]["max_gen"] = tk.StringVar(value=str(default_max))

			self.gui_param_widget_dict[key] = {}

			# Set label for param name
			self.list_label_sv_param[num_column-1].set(label)

			e = ttk.Entry(self.param_frame, textvariable=self.gui_param_dict[key]["min_gen"], justify=tk.CENTER,
						  width=self.entry_text_size)
			e.grid(row=1, column=num_column)
			self.gui_param_widget_dict[key]["e_min"] = e

			e = ttk.Entry(self.param_frame, textvariable=self.gui_param_dict[key]["max_gen"], justify=tk.CENTER,
						  width=self.entry_text_size)
			e.grid(row=2, column=num_column)
			self.gui_param_widget_dict[key]["e_max"] = e

		# There is always the three fisrt column : nb_photon, irf_length and irf_shift
		# Nb photon

		create_value("nb_photon", "nb photon", 1, 1E3, 1E5)
		create_value("noise", "noise (%)", 2, 0, 10)
		create_value("irf_length", "irf length (ps)", 3, 1, 100)
		create_value("irf_shift", "irf shift (ps)",  4, 0, 0)

		i = 0
		for key in self.lifetime_bench.measurement_dummy.params.keys():
			if self.lifetime_bench.measurement_dummy.params[key].user_data is not None:
				if "dontGenerate" in self.lifetime_bench.measurement_dummy.params[key].user_data:
					continue
			param = self.lifetime_bench.measurement_dummy.params[key]
			create_value(key, key, i+5, param.user_data[0], param.user_data[1])
			i += 1


		# Fit param

		self.gui_param_dict_fit = {}
		self.gui_param_widget_dict_fit = {}
		# self.changeModel(None)
		i = 0
		for key in self.lifetime_bench.measurement_dummy.params.keys():
			# Set label for param name
			self.list_label_sv_param_fit[i].set(key)

			# Set current characteritics of the parameter
			param = self.lifetime_bench.measurement_dummy.params[key]
			# value, min, max, b step, hold
			self.gui_param_dict_fit[key] = {}
			self.gui_param_dict_fit[key]["value"] = tk.StringVar(value=str(param.value))
			self.gui_param_dict_fit[key]["min"] = tk.StringVar(value=str(param.min))
			self.gui_param_dict_fit[key]["max"] = tk.StringVar(value=str(param.max))
			self.gui_param_dict_fit[key]["b_step"] = tk.StringVar(value=str(param.brute_step))
			self.gui_param_dict_fit[key]["vary"] = tk.IntVar(value=int(param.vary))

			# Create corresponding widget (entry/button/checkbox)
			self.gui_param_widget_dict_fit[key] = {}
			e = ttk.Entry(self.param_fit_frame, textvariable=self.gui_param_dict_fit[key]["value"], justify=tk.CENTER,
						  width=self.entry_text_size)
			e.grid(row=1 + i, column=1)
			self.gui_param_widget_dict_fit[key]["e_value"] = e

			e = ttk.Entry(self.param_fit_frame, textvariable=self.gui_param_dict_fit[key]["min"], justify=tk.CENTER,
						  width=self.entry_text_size)
			e.grid(row=1 + i, column=4)
			self.gui_param_widget_dict_fit[key]["e_min"] = e

			e = ttk.Entry(self.param_fit_frame, textvariable=self.gui_param_dict_fit[key]["max"], justify=tk.CENTER,
						  width=self.entry_text_size)
			e.grid(row=1 + i, column=5)
			self.gui_param_widget_dict_fit[key]["e_max"] = e

			e = ttk.Entry(self.param_fit_frame, textvariable=self.gui_param_dict_fit[key]["b_step"], justify=tk.CENTER,
						  width=self.entry_text_size)
			e.grid(row=1 + i, column=6)
			self.gui_param_widget_dict_fit[key]["e_b_step"] = e

			# hold check button
			cb = ttk.Checkbutton(self.param_fit_frame, variable=self.gui_param_dict_fit[key]["vary"])
			cb.grid(row=1 + i, column=7)
			self.gui_param_widget_dict_fit[key]["cb_vary"] = cb

			# Set button + and -
			b = tk.Button(master=self.param_fit_frame, text='+',
						  command=lambda: self.value_plus(self.gui_param_dict_fit[key]["value"]))
			b.grid(row=1 + i, column=2)
			self.gui_param_widget_dict_fit[key]["b_plus"] = b
			b = tk.Button(master=self.param_fit_frame, text='-',
						  command=lambda: self.value_minus(self.gui_param_dict_fit[key]["value"]))
			b.grid(row=1 + i, column=3)
			self.gui_param_widget_dict_fit[key]["b_minus"] = b

			i += 1

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

	def get_bench_params(self):
		params = {}
		params["model_name"] = self.cb_model_sv.get()
		params["method1"] = self.cb_method1_sv.get()
		params["method2"] = self.cb_method2_sv.get()
		params["qty_to_min"] = self.cb_minqty_to_min_sv.get()
		params["nb_time_bins"] = int(self.nb_time_bins_sv.get())
		params["time_bins_duration"] = float(self.duration_time_bins_sv.get())
		params["nb_decays"] = int(self.nb_decays_sv.get())
		params["is_ini_guess"] = int(self.check_is_ini_guess_iv.get())
		params["is_fit"] = int(self.check_is_fit_iv.get())
		params["use_error_bar"] = True
		params["lim_fit"] = self.get_lim_for_fit()


		for key in self.lifetime_bench.measurement_dummy.params.keys():
			if self.lifetime_bench.measurement_dummy.params[key].user_data is not None:
				if "dontGenerate" in self.lifetime_bench.measurement_dummy.params[key].user_data:
					continue
			params[key] = {}
			# generator param
			params[key]["min_gen"] = float(self.gui_param_dict[key]["min_gen"].get())
			params[key]["max_gen"] = float(self.gui_param_dict[key]["max_gen"].get())
			#fit param
			params[key]["value"] = float(self.gui_param_dict_fit[key]["value"].get())
			params[key]["min"] = float(self.gui_param_dict_fit[key]["min"].get())
			params[key]["max"] = float(self.gui_param_dict_fit[key]["max"].get())
			params[key]["b_step"] = float(self.gui_param_dict_fit[key]["b_step"].get())
			params[key]["vary"] = bool(self.gui_param_dict_fit[key]["vary"].get())


		for key in ["nb_photon", "noise", "irf_length", "irf_shift"]:
			params[key] = {}
			params[key]["min_gen"] = float(self.gui_param_dict[key]["min_gen"].get())
			params[key]["max_gen"] = float(self.gui_param_dict[key]["max_gen"].get())

		return params



	def update_ui(self):
		pass
		"""
		self.insert_measurement_tree_view(self.burst_measure)
		self.filter_1_graph.replot()
		self.filter_2_graph.replot()
		self.plot()
		"""

	def clear_filter(self):
		self.burst_measure.clear_filter()
		self.update_ui()

	def get_selected_bench_from_treeview(self):
		id_selected_item = self.tree_view.focus()
		parent_iid = self.tree_view.parent(id_selected_item)
		# selected_item = self.tree_view.item(id_selected_item)

		#Have we clicked on a burst measurement or a burst ?
		if parent_iid != "":
			# This is not the main parent node (i.e. user has clicked on a fit, ini_guess or error line
			# We go up one step and work witht the parent id for displaying the curve.
			id_selected_item = parent_iid


		num_burst = 0
		# Burst are retreived via their id() that was stored in self.bench_iid_dict during their insertion
		# items() -> (key, value)
		for id_bench, iid_tk in self.bench_iid_dict.items():
			if iid_tk == id_selected_item:
				for bench in self.lifetime_bench.benchs:
					if id_bench == id(bench):
						return self.lifetime_bench.benchs[num_burst]
					else:
						num_burst += 1
		return None




	def toggle_highlight_bench(self):
		burst = self.get_selected_burst_from_treeview()
		burst.is_highlighted = not burst.is_highlighted
		self.update_ui()


	def toggle_filter_bench(self):
		burst = self.get_selected_burst_from_treeview()
		burst.is_filtered = not burst.is_filtered
		self.update_ui()

	def value_plus(self, tk_string_var):
		# FIXME
		# TODO try execpt
		value = float(tk_string_var.get())
		value *= 1.1
		tk_string_var.set(str(value))

	def value_minus(self, tk_string_var):
		# FIXME
		tk_string_var.set(str(float(tk_string_var.get()) * 0.9))
		value = float(tk_string_var.get())
		value /= 1.1
		tk_string_var.set(str(value))

	def filter_bench(self):

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



	def set_score_statistics(self):

		def format_string(value, nb_digit=2, is_int=False):
			if isinstance(value, str):
				#FIXME
				return

			#FIXME
			return "{:.2f}".format(value).replace(',', ' ')

		self.score_mean_sv.set(format_string(self.lifetime_bench.mean_score))
		self.score_median_sv.set(format_string(self.lifetime_bench.median_score))
		self.score_std_sv.set(format_string(self.lifetime_bench.std_score))

		if self.ax_score_stat is None:
			self.ax_score_stat = self.figure_score_stat.add_subplot(111)

		self.ax_score_stat.clear()
		self.ax_score_stat.bar(self.lifetime_bench.x_hist_score[0:-1], self.lifetime_bench.hist_score)
		self.figure_score_stat.canvas.draw()


	def change_filter1_type(self, event=None):
		type_ = self.cb_value_filter_1_sv.get()
		self.filter_1_graph.plot(self.burst_measure, type_)

	def change_filter2_type(self, event=None):
		type_ = self.cb_value_filter_2_sv.get()
		self.filter_1_graph.plot(self.burst_measure, type_)


	def clear_treeview(self):
		self.tree_view.delete(*self.tree_view.get_children())

	def update_column_name_treeview(self):
		labels = []
		nb_param_fitted = 0
		for key in self.lifetime_bench.benchs[0].fitted_params_dict.keys():
			labels.append(key)
			nb_param_fitted += 1

		for key in self.lifetime_bench.benchs[0].other_fitted_params_dict.keys():
			labels.append(key)
			nb_param_fitted += 1

		for i in range(10 - nb_param_fitted):
			labels.append("")

		self.tree_view.heading("p1", text=labels[0])
		self.tree_view.heading("p2", text=labels[1])
		self.tree_view.heading("p3", text=labels[2])
		self.tree_view.heading("p4", text=labels[3])
		self.tree_view.heading("p5", text=labels[4])
		self.tree_view.heading("p6", text=labels[5])
		self.tree_view.heading("p7", text=labels[6])
		self.tree_view.heading("p8", text=labels[7])
		self.tree_view.heading("p9", text=labels[8])
		self.tree_view.heading("p10", text=labels[9])


	def insert_benchmarks_treeview(self):
		self.clear_treeview()
		for bench in self.lifetime_bench.benchs:
			self.insert_single_bench_treeview(bench)


	def insert_single_bench_treeview(self, bench):
		empty_string = " "
		#Determining p1 and such
		generated_value = []
		fitted_value = []
		rel_error = []
		ini_guess = []

		# self.other_fitted_params_dict[param_key]

		# keys_in_order=[]
		nb_param_fitted = 0
		for key in bench.fitted_params_dict.keys():
			generated_value.append(bench.fitted_params_dict[key])
			fitted_value.append(bench.measurement.fit_results.params[key].value)
			# For value that have not been fitted because the vary option was unset
			if key in bench.measurement.fit_results.init_values:
				ini_guess.append(bench.measurement.fit_results.init_values[key])
			else:
				ini_guess.append(empty_string)
			rel_error.append(bench.errors[key])
			# keys_in_order.append(key)
			nb_param_fitted += 1

		for key in bench.other_fitted_params_dict.keys():
			generated_value.append("")
			fitted_value.append(bench.measurement.fit_results.params[key].value)
			# For value that have not been fitted because the vary option was unset
			if key in bench.measurement.fit_results.init_values:
				ini_guess.append(bench.measurement.fit_results.init_values[key])
			else:
				ini_guess.append(empty_string)
			rel_error.append(empty_string)
			# keys_in_order.append(key)
			nb_param_fitted += 1


		for i in range(10-nb_param_fitted):
			generated_value.append(empty_string)
			fitted_value.append(empty_string)
			ini_guess.append(empty_string)
			rel_error.append(empty_string)


		def format_string(value, nb_digit=2, is_int=False):
			if isinstance(value, str):
				#FIXME
				return

			#FIXME
			return "{:.2f}".format(value).replace(',', ' ')
			if is_int == False:
				return "{:." + str(nb_digit) + "f}".format(value).replace(',', ' ')
			else:
				return "{:." + str(nb_digit) + "f}".format(value).replace(',', ' ')

		i = 0
		for i in range(len(generated_value)):
			generated_value[i] = format_string(generated_value[i])
			ini_guess[i] = format_string(ini_guess[i])
			fitted_value[i] = format_string(fitted_value[i])
			rel_error[i] = format_string(rel_error[i])

		#TODO red color if chi² too high

		# iid = self.tree_view.insert(parent="", index='end', text=exp.file_name)
		# ("num", "r. chi2", "nb photon", "noise", "irf length", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10")

		iid = self.tree_view.insert(parent="", index='end', values=[bench.num, format_string(bench.measurement.fit_results.redchi), int(bench.aux_params_dict["nb_photon"]), format_string(bench.aux_params_dict["noise"]), format_string(bench.aux_params_dict["irf_length"]), format_string(bench.score)] + generated_value)
		self.tree_view.item(iid, open=True)


		# we use id() as a single identificator of the burst
		if id(bench) not in self.bench_iid_dict:
			self.bench_iid_dict[id(bench)] = iid

		# Now we insert on the line below the value obtained with the fit
		iid_ini_guess_value = self.tree_view.insert(parent=iid, index='end', values=["ini guess", "", "", "", "", ""] + ini_guess)

		# Now we insert on the line below the value obtained with the fit
		iid_fit_value = self.tree_view.insert(parent=iid, index='end', values=["fitted", "", "", "", "", ""] + fitted_value)


		# Now we insert on the line below the errors in the fit compared to the generated value
		iid_errors = self.tree_view.insert(parent=iid, index='end', values=["error", "", "", "", "", ""] + rel_error)







	def treeview_measurement_select(self, event):
		self.current_bench = self.get_selected_bench_from_treeview()
		self.graph_results.plot(self.current_bench.measurement)
		self.result_fit_text_area.delete('1.0', tk.END)
		self.result_fit_text_area.insert(tk.INSERT, self.current_bench.measurement.fit_results)

	def treeview_sort_column(self, tv, col, reverse):
		l = [(tv.set(k, col), k) for k in tv.get_children('')]
		l.sort(reverse=reverse)

		# rearrange items in sorted positions
		for index, (val, k) in enumerate(l):
			tv.move(k, '', index)

		# reverse sort next time
		tv.heading(col, command=lambda _col=col: self.treeview_sort_column(tv, _col, not reverse))

		if col == "":
			pass
		pass


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




