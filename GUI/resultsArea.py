import tkinter as tk
from tkinter import ttk
import os

import tkinter.scrolledtext as tkst
import numpy as np
from GUI.graph.Graph_Results import Graph_Results

from tkinter import filedialog


class Results_area():
    def __init__(self, master_frame, view, controller, appearenceParam):
        self.master_frame = master_frame
        self.view = view
        self.controller = controller
        self.appearence_param = appearenceParam


    def populate(self):
        # Results from fit -- Text
        self.frame_left = tk.Frame(self.master_frame)
        self.frame_result_text = tk.LabelFrame(self.frame_left, text="Text",
                                               borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_result_text.grid(row=0, column=0)
        self.frame_result_text.pack(side=tk.TOP, fill="both", expand=True)

        self.resultFitTextArea = tkst.ScrolledText(self.frame_result_text, wrap=tk.WORD, width=45, height=20)
        self.resultFitTextArea.pack(side=tk.LEFT, fill="both", expand=True)
        self.resultFitTextArea.insert(tk.INSERT, "Gimme Results !")

        # Command Graph
        self.frame_cmd_graph = tk.LabelFrame(self.frame_left, text="Cmd graph",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_cmd_graph.grid(row=1, column=0, columnspan=2, sticky=tk.NW)
        self.frame_cmd_graph.pack(side=tk.TOP, fill="both", expand=True)

        ttk.Label(self.frame_cmd_graph, text='x:').grid(row=0, column=0)
        self.x_cursor_sv = tk.StringVar()
        ttk.Entry(self.frame_cmd_graph, width=12, justify=tk.CENTER, font=self.appearence_param.font_xy_coordinate, textvariable=self.x_cursor_sv).grid(row=0, column=1)


        ttk.Label(self.frame_cmd_graph, text='y:').grid(row=1, column=0)
        self.y_cursor_sv = tk.StringVar()
        ttk.Entry(self.frame_cmd_graph, width=12, justify=tk.CENTER, font=self.appearence_param.font_xy_coordinate, textvariable=self.y_cursor_sv).grid(row=1, column=1)

        ttk.Button(self.frame_cmd_graph, text="to x Selec", width=15, command=self.zoom_to_x_selec).grid(row=0, column=2)
        ttk.Button(self.frame_cmd_graph, text="Full", width=15, command=self.zoom_full).grid(row=1, column=2)

        # self.frame_result_text.pack(side=tk.LEFT, fill="both", expand=True)

        # Export Graph
        self.frame_export_graph = tk.LabelFrame(self.frame_left, text="Export Graph",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        # self.frame_export_graph.grid(row=1, column=1, columnspan=2, sticky=tk.NW)
        self.frame_export_graph.pack(side=tk.TOP, fill="both", expand=True)

        ttk.Button(self.frame_export_graph, text="Export Data", width=15, command=lambda: self.export("text")).grid(row=0, column=0)
        ttk.Button(self.frame_export_graph, text="Export Script", width=15, command=lambda: self.export("script")).grid(row=0, column=1)
        ttk.Button(self.frame_export_graph, text="Export Image", width=15, command=lambda: self.export("image")).grid(row=0, column=2)

        self.frame_left.pack(side=tk.LEFT, fill="both", expand=True)


        # Graphs
        self.frame_right = tk.Frame(self.master_frame)
        self.frame_analyze_graphs = tk.LabelFrame(self.frame_right, text="Graph",
                                                  borderwidth=self.appearence_param.frameLabelBorderWidth)

        self.frame_analyze_graphs.grid(row=0, column=1, sticky=tk.NSEW)
        self.graph_results = Graph_Results(self.frame_analyze_graphs, self.view, self.controller,
                                           figsize=(15, 6), dpi=100)

        self.frame_right.pack(side=tk.LEFT, fill="both", expand=True)





    def setTextResult(self, text):
        self.resultFitTextArea.delete('1.0', tk.END)
        self.resultFitTextArea.insert(tk.INSERT, text)

    def set_xy_cursor_position(self, x, y):
        if x is not None:
            self.x_cursor_sv.set("{:.3e}".format(x))
        if y is not None:
            self.y_cursor_sv.set("{:.3e}".format(y))

    def zoom_to_x_selec(self):
        self.graph_results.zoom_to_x_selec()

    def zoom_full(self):
        self.graph_results.zoom_full()

    def export(self, mode):
        base_exp_name = os.path.splitext(os.path.basename(self.controller.current_exp.file_name))[0]
        if mode == "text":
            initialfile = base_exp_name + "_" + self.controller.current_measurement.name + ".txt"
        elif mode == "image":
            initialfile = base_exp_name + "_" + self.controller.current_measurement.name + ".png"
        elif mode == "script":
            initialfile = base_exp_name + "_" + self.controller.current_measurement.name + ".py"
        else:
            initialfile = base_exp_name + "_" + self.controller.current_measurement.name + ".txt"
        file_path = filedialog.asksaveasfile(title="Export file name ?", initialdir=self.controller.view.saveDir, initialfile=initialfile)
        if file_path == None or file_path == '':
            return None
        self.controller.export_graph_result(mode, file_path.name)




