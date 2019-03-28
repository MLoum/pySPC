import tkinter as tk
from tkinter import ttk

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
        self.frame_result_text = tk.LabelFrame(self.master_frame, text="Text",
                                               borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_result_text.grid(row=0, column=0)

        self.resultFitTextArea = tkst.ScrolledText(self.frame_result_text, wrap=tk.WORD, width=45, height=20)
        self.resultFitTextArea.pack(side=tk.LEFT, fill="both", expand=True)
        self.resultFitTextArea.insert(tk.INSERT, "Gimme Results !")

        # #Shared by all the "guiForFitOperation"
        # self.idx_lim_for_fit_min_sv = tk.StringVar()
        # self.idx_lim_for_fit_max_sv = tk.StringVar()



        # Graphs
        self.frame_analyze_graphs = tk.LabelFrame(self.master_frame, text="Graph",
                                                  borderwidth=self.appearence_param.frameLabelBorderWidth)

        self.frame_analyze_graphs.grid(row=0, column=1, sticky=tk.NSEW)
        self.graph_results = Graph_Results(self.frame_analyze_graphs, self.view, self.controller,
                                           figsize=(15, 6), dpi=100)


        self.frame_cmd_graph = tk.LabelFrame(self.master_frame, text="Cmd graph",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_cmd_graph.grid(row=1, column=0, columnspan=2, sticky=tk.NW)

        label = ttk.Label(self.frame_cmd_graph, text='x:')
        label.grid(row=0, column=0)
        self.x_cursor_sv = tk.StringVar()
        e = ttk.Entry(self.frame_cmd_graph, width=20, justify=tk.CENTER, textvariable=self.x_cursor_sv)
        e.grid(row=0, column=1)

        label = ttk.Label(self.frame_cmd_graph, text='y:')
        label.grid(row=1, column=0)
        self.y_cursor_sv = tk.StringVar()
        e = ttk.Entry(self.frame_cmd_graph, width=20, justify=tk.CENTER, textvariable=self.y_cursor_sv)
        e.grid(row=1, column=1)

        b = ttk.Button(self.frame_cmd_graph, text="to x Selec", width=15, command=self.zoom_to_x_selec)
        b.grid(row=0, column=2)

        b = ttk.Button(self.frame_cmd_graph, text="Full", width=15, command=self.zoom_full)
        b.grid(row=1, column=2)


        self.frame_export_graph = tk.LabelFrame(self.master_frame, text="Export Graph",
                                             borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_export_graph.grid(row=1, column=1, columnspan=2, sticky=tk.NW)

        b = ttk.Button(self.frame_export_graph, text="Export Data", width=15, command=lambda: self.export("text"))
        b.grid(row=0, column=0)

        b = ttk.Button(self.frame_export_graph, text="Export Script", width=15, command=lambda: self.export("script"))
        b.grid(row=1, column=0)

        b = ttk.Button(self.frame_export_graph, text="Export Image", width=15, command=lambda: self.export("image"))
        b.grid(row=2, column=0)


    def setTextResult(self, text):
        self.resultFitTextArea.delete('1.0', tk.END)
        self.resultFitTextArea.insert(tk.INSERT, text)

    def set_xy_cursor_position(self, x, y):
        # self.x_cursor_sv.set(np.format_float_scientific(x, unique=False, precision=8, exp_digits=1))
        # self.y_cursor_sv.set(np.format_float_scientific(y, unique=False, precision=8, exp_digits=1))
        self.x_cursor_sv.set(str(x))
        self.y_cursor_sv.set(str(y))

    def zoom_to_x_selec(self):
        self.graph_results.zoom_to_x_selec()

    def zoom_full(self):
        self.graph_results.zoom_full()

    def export(self, mode):
        file_path = filedialog.asksaveasfile(title="Export file name ?", initialdir=self.controller.view.saveDir)
        if file_path == None or file_path == '':
            return None

        self.controller.export_graph_result(mode, file_path)




