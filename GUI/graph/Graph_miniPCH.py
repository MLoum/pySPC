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

        self.createCallBacks()
        self.createWidgets()

    def plot(self, PCH):
        if self.ax is None:
            self.mainAx = self.figure.add_subplot(111)
            self.subplot3D = None
        self.ax.clear()

        # # reduce nb of point to 1000 (approximative size in pixel
        # skipsize = int(PCH.nbOfBin / 1000)
        # idx = np.arange(0, len(PCH.data), skipsize)

        self.ax.semilogx(PCH.data, PCH.time_axis)

        self.figure.canvas.draw()


    def onSpanMove(self, xmin, xmax):
        pass

    def onSpanSelect(self, xmin, xmax):
        pass

    def scrollEvent(self, event):
        pass


