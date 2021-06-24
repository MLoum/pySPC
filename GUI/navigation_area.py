import tkinter as tk
from tkinter import ttk
import copy

from GUI.graph.Graph_navigation import Graph_navigation
from GUI.graph.Graph_miniPCH import Graph_miniPCH
from GUI.graph.Graph_timeZoom import Graph_timeZoom

class navigation_area():
    def __init__(self, master_frame, view, controller, appearenceParam):
        self.master_frame = master_frame
        self.view = view
        self.controller = controller
        self.appearenceParam = appearenceParam

        self.is_keep_time_selection = True
        self.is_keep_bin_thre = True

    def populate(self):

        self.frame_display_channel = tk.LabelFrame(self.master_frame, text="Display Selection", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frame_display_channel.pack(side="top", fill="both", expand=True)

        self.is_channel_1_iv = tk.IntVar()
        ttk.Checkbutton(self.frame_display_channel, text="1", variable=self.is_channel_1_iv, command=self.change_detector_selection).grid(row=0, column=0)
        self.is_channel_1_iv.set(1)

        self.is_channel_2_iv = tk.IntVar()
        ttk.Checkbutton(self.frame_display_channel, text="2", variable=self.is_channel_2_iv, command=self.change_detector_selection).grid(row=0, column=1)

        self.is_channel_3_iv = tk.IntVar()
        ttk.Checkbutton(self.frame_display_channel, text="3", variable=self.is_channel_3_iv, command=self.change_detector_selection).grid(row=0, column=2)

        self.is_channel_4_iv = tk.IntVar()
        ttk.Checkbutton(self.frame_display_channel, text="4", variable=self.is_channel_4_iv, command=self.change_detector_selection).grid(row=0, column=3)

        ttk.Button(self.frame_display_channel, text="Change colors", width=15, command=self.change_color_channel).grid(row=0, column=4)


        self.graph_navigation = Graph_navigation(self.master_frame, self.view, self.controller, figsize=(30, 2), dpi=50)

        self.frameTimeZoom = tk.LabelFrame(self.master_frame, text="Time Evolution (zoom)", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameTimeZoom.pack(side="top", fill="both", expand=True)

        self.time_zoom = TimeZoom_gui(self.frameTimeZoom, self.view, self.controller, self.appearenceParam)
        self.time_zoom.populate()

        self.frame_filter = tk.LabelFrame(self.master_frame, text="Macrotime Filtering", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frame_filter.pack(side="top", fill="both", expand=True)

        self.frame_filter_common = tk.LabelFrame(self.frame_filter, text="Common", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frame_filter_common.grid(row=0, column=0)

        #TODO multiple channel ?
        label = ttk.Label(self.frame_filter_common, text='channel :')
        label.grid(row=0, column=0)
        self.num_channel_sv = tk.StringVar()
        e = ttk.Entry(self.frame_filter_common, textvariable=self.num_channel_sv, justify=tk.CENTER, width=7)
        e.grid(row=0, column=1)
        self.num_channel_sv.set('0')

        label = ttk.Label(self.frame_filter_common, text='Replacement :')
        label.grid(row=1, column=0)
        self.analyze_cb_replacement_sv = tk.StringVar()
        cb = ttk.Combobox(self.frame_filter_common, width=25, justify=tk.CENTER, textvariable=self.analyze_cb_replacement_sv, values='')
        # cb.bind('<<ComboboxSelected>>', self.frame_filter_time_selection)
        cb['values'] = ('nothing', 'glue', 'poissonian_noise')
        self.analyze_cb_replacement_sv.set('nothing')
        cb.grid(row=1, column=1)



        self.frame_filter_time_selection = tk.LabelFrame(self.frame_filter, text="Time Selection", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frame_filter_time_selection.grid(row=0, column=1)

        label = ttk.Label(self.frame_filter_time_selection, text='t1(µs)')
        label.grid(row=0, column=0)
        self.filtre_t1_sv = tk.StringVar()
        e = ttk.Entry(self.frame_filter_time_selection, width=20, justify=tk.CENTER, textvariable=self.filtre_t1_sv)
        e.grid(row=0, column=1)

        label = ttk.Label(self.frame_filter_time_selection, text='t2 (µs)')
        label.grid(row=1, column=0)
        self.filtre_t2_sv = tk.StringVar()
        e = ttk.Entry(self.frame_filter_time_selection, width=20, justify=tk.CENTER, textvariable=self.filtre_t2_sv)
        e.grid(row=1, column=1)

        label = ttk.Label(self.frame_filter_time_selection, text='Action')
        label.grid(row=2, column=0)
        self.toggle_button_action_timeselec = ttk.Button(self.frame_filter_time_selection, text="Keep", width=15, command=self.toggle_filter_mode)
        self.toggle_button_action_timeselec.grid(row=2, column=1)

        b = ttk.Button(self.frame_filter_time_selection, text="Filter", width=15, command=self.filter_time_selection)
        b.grid(row=3, column=0, columnspan=2)

        self.frame_filter_bin_and_threshold = tk.LabelFrame(self.frame_filter, text="Bin and Threshold", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frame_filter_bin_and_threshold.grid(row=0, column=2)

        label = ttk.Label(self.frame_filter_bin_and_threshold, text='Threshold')
        label.grid(row=0, column=0)
        self.filter_threshold_sv = tk.StringVar()
        e = ttk.Entry(self.frame_filter_bin_and_threshold, width=20, justify=tk.CENTER, textvariable=self.filter_threshold_sv)
        e.grid(row=0, column=1)

        label = ttk.Label(self.frame_filter_bin_and_threshold, text='Action')
        label.grid(row=1, column=0)
        self.toggle_button_action_threshold = ttk.Button(self.frame_filter_bin_and_threshold, text="Filter above", width=15, command=self.toggle_filter_mode_bin_thre)
        self.toggle_button_action_threshold.grid(row=1, column=1)

        b = ttk.Button(self.frame_filter_bin_and_threshold, text="Filter", width=15, command=self.filter_bin_threshold)
        b.grid(row=2, column=0, columnspan=2)



    def toggle_filter_mode(self):
        if self.is_keep_time_selection:
            self.toggle_button_action_timeselec.config(text='Discard')
            self.is_keep_time_selection = False
        else:
            self.toggle_button_action_timeselec.config(text='Keep')
            self.is_keep_time_selection = True

    def toggle_filter_mode_bin_thre(self):
        if self.is_keep_bin_thre:
            self.toggle_button_action_threshold.config(text='Filter below')
            self.is_keep_bin_thre = False
        else:
            self.toggle_button_action_threshold.config(text='Filter above')
            self.is_keep_bin_thre = True

    def change_detector_selection(self):
        #FIXME 4 channels max ?
        self.view.displayed_channel = []
        if self.is_channel_1_iv.get():
            self.view.displayed_channel.append(0)
        if self.is_channel_2_iv.get():
            self.view.displayed_channel.append(1)
        if self.is_channel_3_iv.get():
            self.view.displayed_channel.append(2)
        if self.is_channel_4_iv.get():
            self.view.displayed_channel.append(3)
        self.controller.update_all(is_full_update=True)

    def change_color_channel(self):
        pass

    def filter_time_selection(self):
        t1_micro = float(self.filtre_t1_sv.get())
        t2_micro = float(self.filtre_t2_sv.get())
        num_channel = int(self.num_channel_sv.get())
        replacement_mode = self.analyze_cb_replacement_sv.get()
        is_keep_time_selection = self.is_keep_time_selection
        self.controller.macrotime_time_selection_filter(t1_micro, t2_micro, num_channel, is_keep_time_selection, replacement_mode)

    def filter_bin_threshold(self):
        num_channel = int(self.num_channel_sv.get())
        replacement_mode = self.analyze_cb_replacement_sv.get()
        threshold = float(self.filter_threshold_sv.get())
        self.controller.macrotime_bin_threshold_filter(num_channel, threshold, self.is_keep_bin_thre, replacement_mode)

    def copyData(self, target):
        """
        We can't change a widget master with Tkintyer, so one way to ove" a widget from one point
        to another in the GUI is to have two instance of the GUI with different master and copy the -> data <- form
        one to the other
        """
        self.graph_navigation.copyData(target.graph_navigation)
        self.time_zoom.copyData(target.timeZoom)

class TimeZoom_gui():
    def __init__(self, masterFrame, view, controller, appearenceParam):
        self.masterFrame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = appearenceParam

    def populate(self):
        #Bloc commande
        self.frameTimeGraphCommand = tk.LabelFrame(self.masterFrame, text="Command",
                                                   borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameTimeGraphCommand.pack(side=tk.LEFT, fill="both", expand=True)

        #TODO debut et fin et seuil PCH auto.

        label = ttk.Label(self.frameTimeGraphCommand, text='start (ms)')
        label.grid(row=0, column=0)
        self.chronoStart_sv = tk.StringVar()
        self.chronoStart = ttk.Entry(self.frameTimeGraphCommand, width=10, justify=tk.CENTER, textvariable=self.chronoStart_sv)
        self.chronoStart.grid(row=0, column=1)

        label = ttk.Label(self.frameTimeGraphCommand, text='end (ms)')
        label.grid(row=1, column=0)
        self.chronoEnd_sv = tk.StringVar()
        self.chronoEnd = ttk.Entry(self.frameTimeGraphCommand, width=10, justify=tk.CENTER, textvariable=self.chronoEnd_sv)
        self.chronoEnd.grid(row=1, column=1)

        #bin size
        label = ttk.Label(self.frameTimeGraphCommand, text='Bin size (µs)')
        label.grid(row=2, column=0)

        self.bin_size_micros_sv = tk.StringVar()
        self.entryBinSize = ttk.Entry(self.frameTimeGraphCommand, width=6, justify=tk.CENTER,
                                      textvariable=self.bin_size_micros_sv, validatecommand = self.updateChronoBinSize)
        # self.entryBinSize.pack(side=tk.LEFT,  validate = 'key', validatecommand = self.vcmd, padx=self.padx, pady=self.pady)
        # TODO validate
        self.entryBinSize.grid(row=2, column=1)
        self.bin_size_micros_sv.set('100')

        self.isChronoAutoScale = tk.IntVar()
        self.ischrono_auto_scale_check_box =  ttk.Checkbutton(self.frameTimeGraphCommand, text="Autoscale ?", variable=self.isChronoAutoScale)
        self.ischrono_auto_scale_check_box.grid(row=3, column=0, columnspan=2)

        label = ttk.Label(self.frameTimeGraphCommand, text='x')
        label.grid(row=4, column=0)
        self.chronoPos_x = tk.StringVar()
        self.chronoStart = ttk.Entry(self.frameTimeGraphCommand, width=10, justify=tk.CENTER, textvariable=self.chronoStart_sv)
        self.chronoStart.grid(row=4, column=1)

        label = ttk.Label(self.frameTimeGraphCommand, text='y')
        label.grid(row=5, column=0)
        self.chronoPos_y = tk.StringVar()
        self.chronoStart = ttk.Entry(self.frameTimeGraphCommand, width=10, justify=tk.CENTER, textvariable=self.chronoStart_sv)
        self.chronoStart.grid(row=5, column=1)

        label = ttk.Label(self.frameTimeGraphCommand, text='deltaX')
        label.grid(row=6, column=0)
        self.chronoPos_dx = tk.StringVar()
        self.chronoStart = ttk.Entry(self.frameTimeGraphCommand, width=10, justify=tk.CENTER, textvariable=self.chronoStart_sv)
        self.chronoStart.grid(row=6, column=1)

        label = ttk.Label(self.frameTimeGraphCommand, text='deltaY')
        label.grid(row=7, column=0)
        self.chronoPos_dy = tk.StringVar()
        self.chronoStart = ttk.Entry(self.frameTimeGraphCommand, width=10, justify=tk.CENTER, textvariable=self.chronoStart_sv)
        self.chronoStart.grid(row=7, column=1)


        ttk.Label(self.frameTimeGraphCommand, text='mouse x').grid(row=8, column=0)
        self.mouse_pos_x = tk.StringVar()
        ttk.Entry(self.frameTimeGraphCommand, width=10, justify=tk.CENTER, textvariable=self.mouse_pos_x).grid(row=8, column=1)


        ttk.Label(self.frameTimeGraphCommand, text='mouse y').grid(row=8, column=2)
        self.mouse_pos_y = tk.StringVar()
        ttk.Entry(self.frameTimeGraphCommand, width=10, justify=tk.CENTER, textvariable=self.mouse_pos_y).grid(row=8, column=3)



        b = ttk.Button(self.frameTimeGraphCommand, text="redraw", width=6, command=self.redraw_time_zoom_graph)
        b.grid(row=9, column=0)

        # mainGraph for time zoom (cf class Graph_timeZoom in Graph_timeZoom.py)
        self.frameTimeGraph = tk.LabelFrame(self.masterFrame, text="Graph", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameTimeGraph.pack(side=tk.LEFT, fill="both", expand=True)

        self.graph_timeZoom = Graph_timeZoom(self.frameTimeGraph, self.view, self.controller,
                                                 figsize=(10, 5), dpi=50)

        #mini PCH
        self.frameMiniPCH= tk.LabelFrame(self.masterFrame, text="PCH", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMiniPCH.pack(side=tk.LEFT, fill="both", expand=True)

        self.graph_miniPCH = Graph_miniPCH(self.frameMiniPCH, self.view, self.controller,
                                                 figsize=(3, 5), dpi=30)


    def updateChronoBinSize(self):
        self.controller.view.timezoom_bin_size_s = float(self.bin_size_micros_sv.get()) / 1E6
        self.controller.update_navigation()

    def redraw_time_zoom_graph(self):
        self.controller.view.timezoom_bin_size_s = float(self.bin_size_micros_sv.get())/1E6
        self.controller.update_navigation()


    def set_xy_cursor_position(self, x, y):
        if x is not None:
            self.mouse_pos_x.set("{:.3e}".format(x))
        if y is not None:
            self.mouse_pos_y.set("{:.3e}".format(y))


    def copyData(self, target):
        """
        ???? TODO document
        :param target:
        :return:
        """
        target.chronoStart_sv.set(self.chronoStart_sv.get())
        target.chronoEnd_sv.set(self.chronoEnd_sv.get())
        target.binSizeMicros_sv.set(self.bin_size_micros_sv.get())
        target.isChronoAutoScale.set(self.isChronoAutoScale.get())
        target.chronoPos_x.set(self.chronoPos_x.get())
        target.chronoPos_y.set(self.chronoPos_y.get())
        target.chronoPos_dx.set(self.chronoPos_dx.get())
        target.chronoPos_dy.set(self.chronoPos_dy.get())
        self.graph_timeZoom.copyData(target.graph_timeZoom)
        self.graph_miniPCH.copyData(target.graph_miniPCH)
