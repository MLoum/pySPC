import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
from scipy.special import factorial
from scipy.stats import poisson

class BurstAnalysis_gui():
    def __init__(self, master_frame, controller, appearence_param, measurement=None):
        self.master_frame = master_frame
        self.controller = controller
        self.appearence_param = appearence_param
        self.measurement = measurement
        self.modelNames = ("Bin and Threshold", "todo")
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

        label = ttk.Label(self.frame_chronogram, text='bin size (ms)')
        label.grid(row=0, column=0)

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

        label = ttk.Label(self.frame_threshold, text='threshold')
        label.grid(row=0, column=0)

        self.threshold_sv = tk.StringVar()
        e = ttk.Entry(self.frame_threshold, textvariable=self.threshold_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=1)
        self.binsize_sv.set("10")

        label = ttk.Label(self.frame_threshold, text='% False Negative')
        label.grid(row=0, column=2)

        self.false_negative_sv = tk.StringVar()
        e = ttk.Entry(self.frame_threshold, textvariable=self.false_negative_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=3)
        self.false_negative_sv.set("1")

        b = ttk.Button(self.frame_threshold, text="Auto Threshold", command=self.auto_threshold)
        b.grid(row=0, column=4)

        self.frame_graph_trh_grid = tk.Frame(self.frame_threshold)
        self.frame_graph_trh_grid.grid(row=1, column=0, columnspan=5)

        self.frame_graph_threshold = tk.Frame(self.frame_graph_trh_grid)
        self.frame_graph_threshold.pack(side="top", fill="both", expand=True)

        self.pch_graph = Graph_PCH_burst(self.frame_graph_threshold, self.controller.view, self.controller, self, figsize=(5,2), dpi=75)

        # Burst Param
        #################

        self.frame_burst_param = tk.LabelFrame(self.frame_detection, text="c) burst parameters",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_burst_param.pack(side="top", fill="both", expand=True)

        label = ttk.Label(self.frame_burst_param, text='Minimum nb of successive bins')
        label.grid(row=0, column=0)

        self.min_succesive_bin_sv = tk.StringVar()
        e = ttk.Entry(self.frame_burst_param, textvariable=self.min_succesive_bin_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=1)
        self.min_succesive_bin_sv.set("2")

        label = ttk.Label(self.frame_burst_param, text='Maxixum nb of successive NOISE bins')
        label.grid(row=0, column=2)

        self.max_succesive_noise_bin_sv = tk.StringVar()
        e = ttk.Entry(self.frame_burst_param, textvariable=self.max_succesive_noise_bin_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=3)
        self.max_succesive_noise_bin_sv.set("1")

        label = ttk.Label(self.frame_burst_param, text='Min nb of photons in burst')
        label.grid(row=0, column=4)

        self.min_nb_photon_sv = tk.StringVar()
        e = ttk.Entry(self.frame_burst_param, textvariable=self.min_nb_photon_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=5)
        self.min_nb_photon_sv.set("100")


        b = ttk.Button(self.frame_burst_param, text="Launch Detection", command=self.launch_detection)
        b.grid(row=0, column=6)


        self.frame_operation = tk.LabelFrame(self.frame_detection, text="d) What to do",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_operation.pack(side="top", fill="both", expand=True)






        # Statistics
        ##############

        self.frame_statistics = tk.LabelFrame(self.frame_detection, text="e) Statistics information",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_statistics.pack(side="top", fill="both", expand=True)

        # Noise

        self.frame_graph_Noise_int_grid = tk.Frame(self.frame_statistics)
        self.frame_graph_Noise_int_grid.grid(row=0, column=0)

        self.frame_graph_Noise_int = tk.Frame(self.frame_graph_Noise_int_grid)
        self.frame_graph_Noise_int.pack(side="top", fill="both", expand=True)

        self.figure_noise_int = plt.Figure(figsize=(2,2), dpi=150)
        self.canvas_noise_int = FigureCanvasTkAgg(self.figure_noise_int, master=self.frame_graph_Noise_int)
        self.canvas_noise_int.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.frame_graph_Noise_duration_grid = tk.Frame(self.frame_statistics)
        self.frame_graph_Noise_duration_grid.grid(row=0, column=1)

        self.frame_graph_Noise_duration = tk.Frame(self.frame_graph_Noise_duration_grid)
        self.frame_graph_Noise_duration.pack(side="top", fill="both", expand=True)

        self.figure_noise_duration = plt.Figure(figsize=(2,2), dpi=150)
        self.canvas_noise_duration = FigureCanvasTkAgg(self.figure_noise_duration, master=self.frame_graph_Noise_duration)
        self.canvas_noise_duration.get_tk_widget().pack(side='top', fill='both', expand=1)

        # Burst

        self.frame_graph_Burst_int_grid = tk.Frame(self.frame_statistics)
        self.frame_graph_Burst_int_grid.grid(row=0, column=2)

        self.frame_graph_Burst_int = tk.Frame(self.frame_graph_Burst_int_grid)
        self.frame_graph_Burst_int.pack(side="top", fill="both", expand=True)

        self.figure_Burst_int = plt.Figure(figsize=(2,2), dpi=150)
        self.canvas_Burst_int = FigureCanvasTkAgg(self.figure_Burst_int, master=self.frame_graph_Burst_int)
        self.canvas_Burst_int.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.frame_graph_Burst_duration_grid = tk.Frame(self.frame_statistics)
        self.frame_graph_Burst_duration_grid.grid(row=0, column=3)

        self.frame_graph_Burst_duration = tk.Frame(self.frame_graph_Burst_duration_grid)
        self.frame_graph_Burst_duration.pack(side="top", fill="both", expand=True)

        self.figure_Burst_duration = plt.Figure(figsize=(2,2), dpi=150)
        self.canvas_Burst_duration = FigureCanvasTkAgg(self.figure_Burst_duration, master=self.frame_graph_Burst_duration)
        self.canvas_Burst_duration.get_tk_widget().pack(side='top', fill='both', expand=1)



        # tests on burst
        ##############

        self.frame_test = tk.LabelFrame(self.frame_detection, text="f) Tests before validation",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_test.pack(side="top", fill="both", expand=True)


        label = ttk.Label(self.frame_test, text='Num burst')
        label.grid(row=0, column=0)

        self.num_burst_sv = tk.StringVar()
        e = ttk.Entry(self.frame_test, textvariable=self.num_burst_sv, justify=tk.CENTER, width=12)
        e.grid(row=0, column=1)
        self.num_burst_sv.set("0")

        b = ttk.Button(self.frame_test, text="previous burst", command=self.previous_burst)
        b.grid(row=1, column=0)

        b = ttk.Button(self.frame_test, text="next burst", command=self.next_burst)
        b.grid(row=1, column=0)

        #graph chrono

        #graph results

        # Validate
        #####################


        self.frame_validate = tk.LabelFrame(self.frame_detection, text="g) validate",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_validate.pack(side="top", fill="both", expand=True)


        # self.createCallBacks()
        # self.createWidgets()

        self.frame_analysis = tk.Frame(self.notebook)
        self.frame_analysis.pack(side="top", fill="both", expand=True)
        self.notebook.add(self.frame_analysis, text='Burst Analysis')



    def bin_signal(self):
        bin_in_ms = int(self.binsize_sv.get())
        bin_in_tick = (bin_in_ms /1000.0) / self.measurement.exp_param.mAcrotime_clickEquivalentIn_second
        data = self.controller.current_exp.data
        timestamps = data.channels[self.measurement.num_channel].photons['timestamps']
        self.PCH = self.measurement.bin(timestamps, bin_in_tick)

        #Display PCH on the dedicated graph.
        self.pch_graph.plot(self.PCH)


    def change_model(self):
        pass

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
        self.threshold_sv.set(str(chi))
        self.pch_graph.plot(self.PCH)

    def launch_detection(self):
        threshold = int(self.threshold_sv.get())
        min_succesive_bin = int(self.min_succesive_bin_sv.get())
        max_succesive_noise_bin = int(self.max_succesive_noise_bin_sv.get())
        min_nb_photon = int(self.min_nb_photon_sv.get())

        self.measurement.threshold(threshold, min_succesive_bin, max_succesive_noise_bin, min_nb_photon)

        # Display burst infos

    def next_burst(self):
        # plcer des barres/lignes sur les stats
        pass

    def previous_burst(self):
        pass


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

        self.figure.canvas.draw()

    def button_press_event(self, event):
        if event.button == 1:
            self.threshold = int(event.xdata)
            self.burst_GUI.threshold_sv.set(str(self.threshold))
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


