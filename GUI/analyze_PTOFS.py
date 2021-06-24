import tkinter as tk
from tkinter import ttk
from .guiForFitOperation import guiForFitOperation


class guiForFitOperation_PTOFS(guiForFitOperation):

    def __init__(self, master_frame, controller, model_names, nb_param_fit):
        super().__init__(master_frame, controller, model_names, nb_param_fit, fitModeName="PTOFS")

class PTOFS_Analyze_gui():
    def __init__(self, masterFrame, controller, appearence_param, measurement=None):
        self.masterFrame = masterFrame
        self.controller = controller
        self.view = self.controller.view
        self.appearence_param = appearence_param
        self.measurement = measurement
        self.is_keep_selection_for_filter = True
        self.type = "PTOFS"


    def populate(self):
        self.frame_calibration = tk.LabelFrame(self.masterFrame, text="Calibration",
                                           borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_Convert = tk.LabelFrame(self.masterFrame, text="Convert",
                                           borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_filter = tk.LabelFrame(self.masterFrame, text="Filter",
                                          borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_overlay = tk.LabelFrame(self.masterFrame, text="Overleay",
        #                                   borderwidth=self.appearence_param.frameLabelBorderWidth)

        self.frame_fit = tk.LabelFrame(self.masterFrame, text="Fit",
                                       borderwidth=self.appearence_param.frameLabelBorderWidth)

        self.frame_calibration.pack(side="left", fill="both", expand=True)
        self.frame_Convert.pack(side="left", fill="both", expand=True)
        self.frame_filter.pack(side="left", fill="both", expand=True)
        # self.frame_overlay.pack(side="left", fill="both", expand=True)
        self.frame_fit.pack(side="left", fill="both", expand=True)

        # TODO default value object in view


        tk.Label(self.frame_calibration, text="Fiber eq length (m) :").grid(row=0, column=0)
        self.fiber_length_sv = tk.StringVar()
        w = ttk.Entry(self.frame_calibration, textvariable=self.fiber_length_sv, justify=tk.CENTER, width=15)
        # w.bind('<Return>', self.change_IR)
        self.fiber_length_sv.set(100)
        w.grid(row=0, column=1)

        tk.Label(self.frame_calibration, text="micro time calib").grid(row=1, column=0)
        self.microtime_calib_sv = tk.StringVar()
        w = ttk.Entry(self.frame_Convert, textvariable=self.microtime_calib_sv, justify=tk.CENTER, width=15)
        # w.bind('<Return>', self.change_IR)
        self.microtime_calib_sv.set(1000)
        w.grid(row=1, column=1)

        tk.Label(self.frame_calibration, text="wl calib (nm)").grid(row=1, column=2)
        self.wl_calib_sv = tk.StringVar()
        w = ttk.Entry(self.frame_Convert, textvariable=self.wl_calib_sv, justify=tk.CENTER, width=15)
        # w.bind('<Return>', self.change_IR)
        self.wl_calib_sv.set(500)
        w.grid(row=1, column=3)

        tk.Label(self.frame_calibration, text="wl min (nm)").grid(row=1, column=2)
        self.wl_min_sv = tk.StringVar()
        w = ttk.Entry(self.frame_Convert, textvariable=self.wl_calib_sv, justify=tk.CENTER, width=15)
        # w.bind('<Return>', self.change_IR)
        self.wl_calib_sv.set(400)
        w.grid(row=1, column=3)

        tk.Label(self.frame_calibration, text="wl max (nm)").grid(row=1, column=2)
        self.wl_max_sv = tk.StringVar()
        w = ttk.Entry(self.frame_Convert, textvariable=self.wl_calib_sv, justify=tk.CENTER, width=15)
        # w.bind('<Return>', self.change_IR)
        self.wl_calib_sv.set(850)
        w.grid(row=1, column=3)

        ttk.Label(self.frame_Convert, text='Plot : ').grid(row=0, column=0)

        vals = ['m', 's']
        etiqs = ['microtime', 'spectrum']
        self.radio_micro_spec_sv = tk.StringVar()
        self.radio_micro_spec_sv.set(vals[1])
        for i in range(2):
            b = ttk.Radiobutton(self.frame_Convert, variable=self.radio_micro_spec_sv, text=etiqs[i], value=vals[i])
            b.grid(row=0, column=1+i)


        self.is_overlay_on_time_zoom = tk.IntVar(value=0)
        ttk.Checkbutton(self.frame_Convert, text="Overlay on time zoom", variable=self.is_overlay_on_time_zoom, command=self.update_navigation).grid(row=1, column=0, columnspan=3)
        ttk.Button(self.frame_Convert, text="Graph", width=12, command=self.plot_ptofs).grid(row=2, column=0, columnspan=3)


        #Filter
        self.toggle_button = ttk.Button(self.frame_filter, text="Keep selection", width=15, command=self.toggle_filter_mode)
        self.toggle_button.grid(row=0, column=0)

        ttk.Button(self.frame_filter, text="Filter", width=6, command=self.microtime_filter).grid(row=1, column=0)


        # Fit GUI
        self.gui_for_fit_operation = guiForFitOperation_PTOFS(self.frame_fit, self.controller, ["None"], nb_param_fit=8)
        self.gui_for_fit_operation.populate()

    def plot_ptofs(self):
        self.controller.calculate_measurement()
        self.controller.graph_measurement()

    def update_navigation(self):
        self.controller.update_navigation()

    def toggle_filter_mode(self):
        if self.is_keep_selection_for_filter:
            self.toggle_button.config(text='Filter selection')
            self.is_keep_selection_for_filter = False
        else:
            self.toggle_button.config(text='Keep selection')
            self.is_keep_selection_for_filter = True

    def microtime_filter(self):
        # FIXME MultiChannel
        # num_channel = int(self.num_channel_sv.get())
        num_channel = 0
        self.controller.microtime_filter(num_channel, self.is_keep_selection_for_filter)

