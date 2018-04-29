import tkinter as tk
from tkinter import ttk
import copy

from GUI.graph.Graph_navigation import  Graph_navigation
from GUI.graph.interactiveGraphs import Graph_miniPCH
from GUI.graph.Graph_timeZoom import  Graph_timeZoom

class navigation_area():
    def __init__(self, masterFrame, view, controller, appearenceParam):
        self.masterFrame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = appearenceParam

    def populate(self):
        self.graph_navigation = Graph_navigation(self.masterFrame, self.view, self.controller, figsize=(15, 2), dpi=50)

        self.frameTimeZoom = tk.LabelFrame(self.masterFrame, text="Time Evolution (zoom)", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameTimeZoom.pack(side="top", fill="both", expand=True)

        self.timeZoom = TimeZoom_gui(self.frameTimeZoom, self.view, self.controller, self.appearenceParam)
        self.timeZoom.populate()

    def copyData(self, target):
        """
        We can't change a widget master with Tkintyer, so one way to ove" a widget from one point
        to another in the GUI is to have two instance of the GUI with different master and copy the -> data <- form
        one to the other
        """
        self.graph_navigation.copyData(target.graph_navigation)
        self.timeZoom.copyData(target.timeZoom)

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

        self.binSizeMicros_sv = tk.StringVar()
        self.entryBinSize = ttk.Entry(self.frameTimeGraphCommand, width=6, justify=tk.CENTER,
                                      textvariable=self.binSizeMicros_sv, validatecommand = self.updateChronoBinSize)
        # self.entryBinSize.pack(side=tk.LEFT,  validate = 'key', validatecommand = self.vcmd, padx=self.padx, pady=self.pady)
        # TODO validate
        self.entryBinSize.grid(row=2, column=1)
        self.binSizeMicros_sv.set('100')

        self.isChronoAutoScale = tk.IntVar()
        self.ischronoAutoScaleCheckBox =  ttk.Checkbutton(self.frameTimeGraphCommand, text="Autoscale ?", variable=self.isChronoAutoScale)
        self.ischronoAutoScaleCheckBox.grid(row=3, column=0, columnspan=2)

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


        #TODO les autre commandes necessaire. Redraw ?

        # mainGraph for time zoom (cf class Graph_timeZoom in InteractiveGraph.py)
        self.frameTimeGraph = tk.LabelFrame(self.masterFrame, text="Graph", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameTimeGraph.pack(side=tk.LEFT, fill="both", expand=True)

        self.graph_timeZoom = Graph_timeZoom(self.frameTimeGraph, self.view, self.controller,
                                                 figsize=(15, 5), dpi=50)

        #mini PCH
        self.frameMiniPCH= tk.LabelFrame(self.masterFrame, text="PCH", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMiniPCH.pack(side=tk.LEFT, fill="both", expand=True)

        self.graph_miniPCH = Graph_miniPCH(self.frameMiniPCH, self.view, self.controller,
                                                 figsize=(1, 5), dpi=20)


    def updateChronoBinSize(self):
        self.view.currentBinSize_s = float(self.binSizeMicros_sv.get())/1E6
        self.controller.updateNavigation()

    def copyData(self, target):
        target.chronoStart_sv.set(self.chronoStart_sv.get())
        target.chronoEnd_sv.set(self.chronoEnd_sv.get())
        target.binSizeMicros_sv.set(self.binSizeMicros_sv.get())
        target.isChronoAutoScale.set(self.isChronoAutoScale.get())
        target.chronoPos_x.set(self.chronoPos_x.get())
        target.chronoPos_y.set(self.chronoPos_y.get())
        target.chronoPos_dx.set(self.chronoPos_dx.get())
        target.chronoPos_dy.set(self.chronoPos_dy.get())
        self.graph_timeZoom.copyData(target.graph_timeZoom)
        self.graph_miniPCH.copyData(target.graph_miniPCH)
