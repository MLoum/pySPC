import tkinter as tk
from tkinter import ttk


#from pylab import *
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np


from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Cursor
from matplotlib.widgets import MultiCursor

from .interactiveGraphs import InteractiveGraph

class Graph_miniPCH(InteractiveGraph):
    """
    doc todo
    """

    def __init__(self, master_frame, view, controller, figsize, dpi):
        super().__init__(master_frame, view, controller, figsize, dpi)
        #self.ax.axis('off')
        self.figure.tight_layout()

        self.threshold = None
        self.pchs = None

        self.createCallBacks()
        self.createWidgets()

    def plot(self, PCHs):
        self.pchs = PCHs

        # if self.ax is None:
        #     self.mainAx = self.figure.add_subplot(111)
        #     self.subplot3D = None
        self.ax.clear()

        # # reduce nb of point to 1000 (approximative size in pixel
        # skipsize = int(PCH.nbOfBin / 1000)
        # idx = np.arange(0, len(PCH.data), skipsize)

        for num_channel in self.view.displayed_channel:
            pch = PCHs[num_channel]
            self.ax.semilogx(pch.data, pch.time_axis)

        if self.threshold is not None:
            self.ax.hlines(self.threshold, 0, PCHs.data.max(), linewidth=4)

        self.figure.canvas.draw()

    def button_press_event(self, event):
        if event.button == 1:
            # self.threshold = event.ydata
            # self.plot(self.pch)
            self.controller.set_macrotime_filter_threshold(event.ydata)


    def onSpanMove(self, xmin, xmax):
        pass

    def onSpanSelect(self, xmin, xmax):
        pass

    def scrollEvent(self, event):
        pass

    def createWidgets(self):
        # super().createWidgets()
        self.cursor_h = Cursor(self.ax, useblit=True, color='red', horizOn=True, vertOn=False, linewidth=4)

        # self.cursor_h.set_active(False)
        # self.cursor_h.drawon = True
        # drawon
        # eventson
        # self.setOnOffCursors(True)


