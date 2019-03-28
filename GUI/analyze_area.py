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


        # # TreeView
        #
        # self.frame_tree_view = tk.LabelFrame(self.masterFrame, text="List measurements", borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_tree_view.pack(side="top", fill="both", expand=True)
        #
        # #https://riptutorial.com/tkinter/example/31885/customize-a-treeview
        # self.tree_view = ttk.Treeview(self.frame_tree_view)
        # self.tree_view["columns"] = ("name", "type", "comment", "t_start_µs", "t_end_µs")
        # # remove first empty column with the identifier
        # self.tree_view['show'] = 'headings'
        # # tree.column("#0", width=270, minwidth=270, stretch=tk.NO) tree.column("one", width=150, minwidth=150, stretch=tk.NO) tree.column("two", width=400, minwidth=200) tree.column("three", width=80, minwidth=50, stretch=tk.NO)
        # self.tree_view.column("name", width=300)
        # self.tree_view.column("type", width=75)
        # self.tree_view.column("comment", width=600)
        # self.tree_view.column("t_start_µs", width=50)
        # self.tree_view.column("t_end_µs", width=50)
        #
        # self.tree_view.heading("name", text="Name")
        # self.tree_view.heading("type", text="Type")
        # self.tree_view.heading("comment", text="Comment")
        # self.tree_view.heading("t_start_µs", text="t_start_µs")
        # self.tree_view.heading("t_end_µs", text="t_end_µs")
        #
        # ysb = ttk.Scrollbar(self.frame_tree_view, orient='vertical', command=self.tree_view.yview)
        # self.tree_view.grid(row=0, column=0, sticky='nsew')
        # ysb.grid(row=0, column=1, sticky='ns')
        # self.tree_view.configure(yscroll=ysb.set)
        #
        # self.tree_view.bind('<<TreeviewSelect>>', self.treeview_measurement_select)
        #
        # self.tree_view.grid(row=0, column=0)
        #
        # self.frame_tree_view_cmd = tk.Frame(self.frame_tree_view)
        # self.frame_tree_view_cmd.grid(row=1, column=0)
        #
        # b = ttk.Button(self.frame_tree_view_cmd, text="+", width=10, command=self.add_measurement)
        # b.grid(row=0, column=0)
        #
        # b = ttk.Button(self.frame_tree_view_cmd, text="-", width=10, command=self.del_measurement)
        # b.grid(row=0, column=1)
        #
        # b = ttk.Button(self.frame_tree_view_cmd, text="Duplicate", width=10, command=self.duplicate_measurement)
        # b.grid(row=0, column=2)
        #
        #
        # self.frame_identification = tk.LabelFrame(self.masterFrame, text="Information", borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_identification.grid(row=0, column=1)
        #
        # label = ttk.Label(self.frame_identification, text='name')
        # label.grid(row=0, column=0)
        #
        # self.measurement_name_sv = tk.StringVar()
        # e = ttk.Entry(self.frame_identification, textvariable=self.measurement_name_sv, justify=tk.CENTER, width=50)
        # e.grid(row=0, column=1)
        #
        # self.label_type_sv = tk.StringVar()
        # self.label_type = ttk.Label(self.frame_identification, text='type :', textvariable=self.label_type_sv)
        # self.label_type.grid(row=0, column=2)
        #

        # label = ttk.Label(self.frame_identification, text='Comment :')
        # label.grid(row=1, column=0)
        # self.comment_text = tkst.ScrolledText(self.frame_identification, wrap=tk.WORD, width=100, height=5)
        # self.comment_text.grid(row=1, column=1, columnspan=2)


        #
        # # Graph
        # self.frame_common_graph = tk.LabelFrame(self.frame_common, text="Graph command", borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_common_graph.pack(side="top", fill="both", expand=True)
        #
        # label = ttk.Label(self.frame_common_graph, text='x1')
        # label.grid(row=0, column=0)
        #
        # self.x1_graph_sv = tk.StringVar()
        # e = ttk.Entry(self.frame_common_graph, textvariable=self.x1_graph_sv, justify=tk.CENTER, width=7)
        # e.grid(row=0, column=1)
        #
        # label = ttk.Label(self.frame_common_graph, text='x2')
        # label.grid(row=0, column=2)

        # self.x2_graph_sv = tk.StringVar()
        # e = ttk.Entry(self.frame_common_graph, textvariable=self.x2_graph_sv, justify=tk.CENTER, width=7)
        # e.grid(row=0, column=3)
        #
        # label = ttk.Label(self.frame_common_graph, text='y1')
        # label.grid(row=1, column=0)
        #
        # self.y1_graph_sv = tk.StringVar()
        # e = ttk.Entry(self.frame_common_graph, textvariable=self.y1_graph_sv, justify=tk.CENTER, width=7)
        # e.grid(row=1, column=1)
        #
        # label = ttk.Label(self.frame_common_graph, text='y2')
        # label.grid(row=1, column=2)
        #
        # self.y2_graph_sv = tk.StringVar()
        # e = ttk.Entry(self.frame_common_graph, textvariable=self.y2_graph_sv, justify=tk.CENTER, width=7)
        # e.grid(row=1, column=3)
        #
        # b = ttk.Button(self.frame_common_graph, text="to xSelec", width=10, command=self.scale_graph_result_to_x_selec)
        # b.grid(row=0, column=4)
        #
        # b = ttk.Button(self.frame_common_graph, text="Autoscale", width=10, command=self.autoscale_graph_result)
        # b.grid(row=1, column=4)
        #
        # b = ttk.Button(self.frame_common_graph, text="Export", width=10, command=self.export_graph_result)
        # b.grid(row=0, column=5)


        self.frame_operation = tk.LabelFrame(self.master_frame, text="Operation", borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_operation.pack(side="top", fill="both", expand=True)


        self.frameResult = tk.LabelFrame(self.master_frame, text="Results", borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frameResult.pack(side="top", fill="both", expand=True)

        self.resultArea_gui = Results_area(self.frameResult, self.view, self.controller, self.appearence_param)
        self.resultArea_gui.populate()



    def display_measurement(self, measurement):
        for child in self.frame_operation.winfo_children():
            child.destroy()

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
            if measurement.IR_raw is not None:
                self.analyze_gui.isDraw_IR.set(measurement.use_IR)
                self.analyze_gui.ir_name_sv.set(measurement.IR_name)
                self.analyze_gui.ir_start_sv.set(str(measurement.IR_start))
                self.analyze_gui.ir_end_sv.set(str(measurement.IR_end))
                self.analyze_gui.shiftIR_amount_sv.set(str(measurement.IR_shift))
                self.analyze_gui.bckg_IR_sv.set(str(measurement.IR_bckg))

            self.gui_for_fit_operation = self.analyze_gui.gui_for_fit_operation
        elif measurement.type == "DLS":
            self.gui_for_fit_operation = None
        elif measurement.type == "PCH":
            self.gui_for_fit_operation = None

        self.frame_operation.pack(side="top", fill="both", expand=True)

    def copyData(self, target):
        """
        We can't change a widget master with Tkintyer, so one way to ove" a widget from one point
        to another in the GUI is to have two instance of the GUI with different master and copy the -> data <- form
        one to the other
        """
        #TODO
        pass
        #self.resultArea_gui.graph_results.copyData(target.graph_navigation)

    def add_measurement(self):
        if self.controller.current_exp is not None:
            d = add_measurement_dialog(self.master_frame, "add experiment", self.controller)
            if d.result is not None:
                exp_type, exp_name, exp_comment = d.result
                new_measurement = self.controller.create_measurement(exp_type, exp_name, exp_comment)
                self.controller.add_measurement(new_measurement)
                #TODO tick start and end.
                #TODO node for each file and multi_measurement.
                iid = self.tree_view.insert(parent="", index='end', values=(exp_name, exp_type, exp_comment, 0, -1))
                self.tree_view.focus(iid)
                self.treeview_measurement_select(None)
        else:
            self.controller.log_message("No experiment loaded")

    def del_measurement(self):
        list_selection = self.list_measurement.curselection()
        num = int(list_selection[0])
        measurement_name = self.list_measurement.get(num)
        self.controller.del_measurement(measurement_name)
        self.list_measurement.delete(num)

    def duplicate_measurement(self):
        list_selection = self.list_measurement.curselection()
        num = int(list_selection[0])
        measurement_name = self.list_measurement.get(num)
        self.controller.duplicate_measurement(measurement_name)
        self.list_measurement.insert(tk.END, measurement_name + "_b")

    def treeview_measurement_select(self, event):
        id_selected_item = self.tree_view.focus()
        selected_item = self.tree_view.item(id_selected_item)
        measurement_name = selected_item["values"][0]
        measurement_type = selected_item["values"][1]
        measurement = self.controller.set_current_measurement(measurement_name)
        #display corresponding operation frame



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


    # def on_ChangeAnalyzeTab(self, event):
    #     if self.view.is_a_FileLoaded == False:
    #         return
    #     currentNumTab  =  self.tabOperation.index(self.tabOperation.select())
    #     if currentNumTab == 0:
    #         self.view.currentOperation = "macro"
    #     elif currentNumTab == 1:
    #         self.view.currentOperation = "micro"
    #     elif currentNumTab == 2:
    #         self.view.currentOperation = "filter"
    #     elif currentNumTab == 3:
    #         self.view.currentOperation = "FCS"
    #     elif currentNumTab == 4:
    #         self.view.currentOperation = "DLS"
    #     # self.view.controller.update_analyze()

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
