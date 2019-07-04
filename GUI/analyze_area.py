import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import tkinter.scrolledtext as tkst

from .analyze_Lifetime import lifeTimeAnalyze_gui
from .analyze_DLS import DLS_Analyze_gui
from .analyze_FCS import FCS_Analyze_gui

from .resultsArea import Results_area

class Analyze_area():
    def __init__(self, masterFrame, view, controller, appearenceParam):
        self.master_frame = masterFrame
        self.view = view
        self.controller = controller
        self.appearence_param = appearenceParam

        self.analyze_gui = None
        self.gui_for_fit_operation = None

    def populate(self):
        # self.frame_common = tk.LabelFrame(self.masterFrame, text="Common", borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_common.pack(side="top", fill="both", expand=True)
        #
        # self.frame_common_cmd = tk.LabelFrame(self.frame_common, text="cmd", borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_common_cmd.pack(side="top", fill="both", expand=True)

        self.frame_selection = tk.LabelFrame(self.master_frame, text="Selection", borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_selection.pack(side="top", fill="both", expand=True)

        #combo box analyse Source
        label = ttk.Label(self.frame_selection, text='Source')
        label.grid(row=0, column=0)
        self.analyze_combo_box_source_sv = tk.StringVar()
        cb = ttk.Combobox(self.frame_selection, width=25, justify=tk.CENTER, textvariable=self.analyze_combo_box_source_sv, values='')
        cb.bind('<<ComboboxSelected>>', self.change_analyze_source)
        cb['values'] = ('Whole', 'Time Zoom', 'Selection')
        self.analyze_combo_box_source_sv.set('Time Zoom')
        cb.grid(row=0, column=1)

        #ProgressBar
        """
        If your program cannot accurately depict the relative progress that this widget is supposed to display, use mode='indeterminate'. In this mode, a rectangle bounces back and forth between the ends of the widget once you use the .start() method.
        If your program has some measure of relative progress, use mode='determinate'. In this mode, your program can move the indicator to a specified position along the widget's track.
        """
        self.analyzePgb = ttk.Progressbar(self.frame_selection, orient="horizontal", length=200, mode='indeterminate')
        self.analyzePgb.grid(row=0, column=2)
        # #self.analyzePgb.ste
        #
        #Live Analysis ?
        self.is_live_int_var = tk.IntVar()
        self.is_live_check_box = ttk.Checkbutton(self.frame_selection, text="Live ?", variable=self.is_live_int_var)
        self.is_live_check_box.grid(row=0, column=3)

        self.frame_operation = tk.LabelFrame(self.master_frame, text="Operation", borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_operation.pack(side="top", fill="both", expand=True)


        self.frameResult = tk.LabelFrame(self.master_frame, text="Results", borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frameResult.pack(side="top", fill="both", expand=True)

        self.resultArea_gui = Results_area(self.frameResult, self.view, self.controller, self.appearence_param)
        self.resultArea_gui.populate()



    def display_measurement(self, measurement):
        for child in self.frame_operation.winfo_children():
            child.destroy()

        if measurement is None:
            self.gui_for_fit_operation = None
            return

        if measurement.type == "FCS":
            self.analyze_gui = FCS_Analyze_gui(self.frame_operation, self.controller,
                                                       self.appearence_param, measurement)
            self.analyze_gui.populate()
            self.gui_for_fit_operation = self.analyze_gui.gui_for_fit_operation
        elif measurement.type == "chronogram":
            pass
        elif measurement.type == "lifetime":
            self.analyze_gui = lifeTimeAnalyze_gui(self.frame_operation, self.controller, self.appearence_param, measurement)
            self.analyze_gui.populate()
            if measurement.IRF is not None and measurement.IRF.raw_data is not None:
                self.analyze_gui.isDraw_IR.set(measurement.use_IR)
                self.analyze_gui.ir_start_sv.set(str(measurement.IRF.start))
                self.analyze_gui.ir_end_sv.set(str(measurement.IRF.end))
                # self.analyze_gui.shiftIR_amount_sv.set(str(measurement.IRF.shift))
                self.analyze_gui.bckg_IR_sv.set(str(measurement.IRF.bckgnd))

            self.gui_for_fit_operation = self.analyze_gui.gui_for_fit_operation
        elif measurement.type == "DLS":
            self.gui_for_fit_operation = None
        elif measurement.type == "PCH":
            self.gui_for_fit_operation = None

        self.frame_operation.pack(side="top", fill="both", expand=True)



    def set_current_measurement(self, measurement):
        if measurement.type == "FCS":
            # grid_forget
            self.FCS_gui = FCS_Analyze_gui(self.frame_operation, self.controller,
                                                       self.appearence_param)
            self.FCS_gui.populate()
        elif measurement.type == "lifetime":
            self.life_time_analyze_gui = lifeTimeAnalyze_gui(self.frame_operation, self.controller, self.appearence_param)
            self.life_time_analyze_gui.populate()
        elif measurement.type == "DLS":
            pass


    def change_analyze_source(self, event=None):
        # Moved to the controller
        pass

    def scale_graph_result_to_x_selec(self):
        self.controller.replot_result(is_zoom_x_selec=True, is_autoscale=False)

    def autoscale_graph_result(self):
        self.controller.replot_result(is_zoom_x_selec=False, is_autoscale=True)

    def export_graph_result(self):
        file_path = filedialog.asksaveasfile(title="Export Graph")
        if file_path == None or file_path == '':
            return None

        self.controller.export_graph_result(mode="text", file_path=file_path)


# #https://stackoverflow.com/questions/673174/file-dialogs-of-tkinter-in-python-3
# from tkinter import filedialog, messagebox, simpledialog
#
#
# class add_measurement_dialog(simpledialog.Dialog):
#     """
#     http://effbot.org/tkinterbook/tkinter-dialog-windows.htm
#     """
#     def __init__(self, master, title, controller):
#         self.controller = controller
#         super().__init__(master, title)
#
#     def body(self, master):
#
#         ttk.Label(master, text="Type:").grid(row=0)
#         ttk.Label(master, text="Name :").grid(row=1)
#         ttk.Label(master, text="Comment :").grid(row=2)
#
#         self.cb_sv = tk.StringVar()
#         self.cb = ttk.Combobox(master, width=25, justify=tk.CENTER, textvariable=self.cb_sv, values='')
#         self.cb['values'] = ('FCS', 'lifetime', 'DLS')
#         self.cb.set('FCS')
#         self.cb.bind('<<ComboboxSelected>>', self.change_type)
#         self.cb.grid(row=0, column=1)
#
#         self.e1_svar = tk.StringVar()
#         self.e1 = ttk.Entry(master, textvariable=self.e1_svar)
#         self.e2 = ttk.Entry(master)
#
#         #default value
#         default_type = "FCS"
#         self.e1.insert(tk.END, default_type)
#         self.e2.insert(tk.END, self.controller.get_available_name_for_measurement(default_type))
#
#         self.e1.grid(row=1, column=1)
#         self.e2.grid(row=2, column=1)
#
#         self.result = None
#         return self.e2 # initial focus
#
#     def apply(self):
#         first = self.cb_sv.get()
#         second = self.e1.get()
#         third = self.e2.get()
#         self.result = first, second, third
#
#     def change_type(self, event=None):
#         self.e1_svar.set(self.controller.get_available_name_for_measurement(self.cb_sv.get()))
