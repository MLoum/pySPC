import tkinter as tk
from tkinter import ttk


#from pylab import *
import matplotlib.pyplot as plt
from IPython import embed
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
#import midiControl
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Cursor
from matplotlib.widgets import MultiCursor

from .interactiveGraphs import InteractiveGraph

class Graph_Results(InteractiveGraph):
    """
    doc todo
    """

    def __init__(self, masterFrame, view, controller, figsize, dpi):
        #super().__init__(masterFrame, view, controller, figsize, dpi)


        self.masterFrame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = view.appearenceParam

        self.x_selec_min = None
        self.x_selec_max = None

        self.x_selec_min_idx = None
        self.x_selec_max_idx = None

        self.ctrl_is_held = False
        self.shift_is_held = False

        self.type = None

        self.frame= tk.Frame(self.masterFrame)
        self.frame.pack(side="top", fill="both", expand=True)

        self.figure = plt.Figure(figsize=figsize, dpi=dpi)
        #[left, bottom, width, height]
        self.ax = self.figure.add_axes([0.08, 0.3, 0.9, 0.65],
                           xticklabels=[])
        self.axResidual = self.figure.add_axes([0.08, 0.1, 0.9, 0.25])

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)


        #self.ax.axis('off')
        #self.figure.tight_layout()

        self.createCallBacks()
        self.createWidgets()

    def plot(self, type, data):
        self.type = type
        self.data = data
        self.ax.clear()
        self.axResidual.clear()

        if type == "lifetime":
            # data is of type "lifeTimeMeasurements" (cf correspondinf class in core/analyse)
            lf = data

            y = lf.data
            x = lf.timeAxis

            self.ax.set_xlabel("Microtime")
            self.ax.set_ylabel("Intensity")
            self.axResidual.set_xlabel("Microtime")
            self.axResidual.set_ylabel("Residual")

            self.ax.set_xlim(x[0], x[-1])

            #NB : semilogy

            if self.x_selec_min is None :
                self.ax.plot(x, y)
            else :
                x1 = np.searchsorted(x, self.x_selec_min)
                x2 = np.searchsorted(x, self.x_selec_max)
                self.ax.plot(x[:x1], y[:x1], alpha=self.appearenceParam.alphaNonSelectedGraph)
                self.ax.plot(x[x1:x2], y[x1:x2], alpha=1)
                self.ax.plot(x[x2:], y[x2:], alpha=self.appearenceParam.alphaNonSelectedGraph)

            self.axResidual.plot(x, np.zeros(np.size(y)))
            self.figure.canvas.draw()

        elif type == "FCS":
            FCS_Measurements = data

            y = FCS_Measurements.data
            x = FCS_Measurements.timeAxis

            self.ax.set_xlim(x[0], x[-1])

            if self.x_selec_min is None :
                self.ax.semilogx(x, y)
            else :
                x1 = np.searchsorted(x, self.x_selec_min)
                x2 = np.searchsorted(x, self.x_selec_max)
                self.ax.semilogx(x[:x1], y[:x1], alpha=self.appearenceParam.alphaNonSelectedGraph)
                self.ax.semilogx(x[x1:x2], y[x1:x2], alpha=1)
                self.ax.semilogx(x[x2:], y[x2:], alpha=self.appearenceParam.alphaNonSelectedGraph)

            self.axResidual.plot(x, np.zeros(np.size(y)))
            self.figure.canvas.draw()

        elif type == "DLS":
            DLS_Measurements = data

            y = DLS_Measurements.data
            x = DLS_Measurements.timeAxis

            self.ax.set_xlim(x[0], x[-1])

            if self.x_selec_min is None :
                self.ax.semilogx(x, y)
            else :
                # x1 = np.searchsorted(x, self.x_selec_min)
                # x2 = np.searchsorted(x, self.x_selec_max)
                x1 = DLS_Measurements.idxStart
                x2 = DLS_Measurements.idxEnd
                self.ax.semilogx(x[:x1], y[:x1], alpha=self.appearenceParam.alphaNonSelectedGraph)
                self.ax.semilogx(x[x1:x2], y[x1:x2], alpha=1)
                self.ax.semilogx(x[x2:], y[x2:], alpha=self.appearenceParam.alphaNonSelectedGraph)

            self.axResidual.plot(x, np.zeros(np.size(y)))
            self.figure.canvas.draw()

    def plotFit(self, type, data):
        self.type = type
        self.ax.clear()
        self.axResidual.clear()
        if type == "lifetime":
            # data is of type "lifeTimeMeasurements" (cf correspondinf class in core/analyse)
            lf = data

            self.ax.set_xlabel("Microtime (ns)")
            self.ax.set_ylabel("Intensity")
            self.axResidual.set_xlabel("Microtime")
            self.axResidual.set_ylabel("Residual")

            y = lf.data
            x = lf.timeAxis

            #TODO ZOOM
            self.ax.set_xlim(x[0], x[-1])
            self.axResidual.set_xlim(x[0], x[-1])

            if self.x_selec_min is None :
                # (Re)plot data
                self.ax.plot(x, y)

                # Plot fit
                self.ax.set_xlim(lf.eval_x_axis[0], lf.eval_x_axis[-1])
                self.ax.plot(lf.eval_x_axis, lf.eval_y_axis, self.appearenceParam.lineTypeFitLifetime)

                self.axResidual.plot(lf.eval_x_axis,
                                     lf.residuals)

            else :
                #(Re)plot data
                # x1 = np.searchsorted(x, self.x_selec_min)
                # x2 = np.searchsorted(x, self.x_selec_max)
                x1 = lf.idxStart
                x2 = lf.idxEnd
                self.ax.plot(x[:x1], y[:x1], alpha=self.appearenceParam.alphaNonSelectedGraph)
                self.ax.plot(x[x1:x2], y[x1:x2], alpha=1)
                self.ax.plot(x[x2:], y[x2:], alpha=self.appearenceParam.alphaNonSelectedGraph)

                # Plot fit
                self.ax.plot(lf.eval_x_axis, lf.eval_y_axis, self.appearenceParam.lineTypeFitLifetime)

                self.axResidual.plot(lf.eval_x_axis,
                                     lf.residuals)
            self.figure.canvas.draw()

        if type == "DLS":
            # data is of type "lifeTimeMeasurements" (cf correspondinf class in core/analyse)
            DLS_Measurements = data

            y = DLS_Measurements.data
            x = DLS_Measurements.timeAxis

            self.ax.set_xlim(x[0], x[-1])
            self.axResidual.set_xlim(x[0], x[-1])

            # self.ax.plot(lf.timeAxis, lf.data)

            if self.x_selec_min is None:
                # (Re)plot data
                self.ax.semilogx(x, y)

                # Plot fit
                self.ax.set_xlim(x[0], x[-1])
                self.ax.semilogx(lf.eval_x_axis, lf.eval_y_axis, self.appearenceParam.lineTypeFitLifetime)

                self.axResidual.semilogx(lf.eval_x_axis,
                                     lf.residuals)

            else:
                # (Re)plot data
                # x1 = np.searchsorted(x, self.x_selec_min)
                # x2 = np.searchsorted(x, self.x_selec_max)
                x1 = DLS_Measurements.idxStart
                x2 = DLS_Measurements.idxEnd
                self.ax.semilogx(x[:x1], y[:x1], alpha=self.appearenceParam.alphaNonSelectedGraph)
                self.ax.semilogx(x[x1:x2], y[x1:x2], alpha=1)
                self.ax.semilogx(x[x2:], y[x2:], alpha=self.appearenceParam.alphaNonSelectedGraph)

                # Plot fit
                self.ax.semilogx(DLS_Measurements.eval_x_axis, DLS_Measurements.eval_y_axis, self.appearenceParam.lineTypeFitLifetime)

                self.axResidual.semilogx(DLS_Measurements.eval_x_axis,
                                         DLS_Measurements.residuals)

            self.figure.canvas.draw()




    def onSpanMove(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        # self.view.currentTimeWindow = [xmin, xmax]
        # self.controller.zoom()

    def onSpanSelect(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        self.x_selec_min = xmin
        self.x_selec_max = xmax
        #Update GUI
        self.controller.setLimX_fit(self.x_selec_min, self.x_selec_max)
        self.plot(self.type, self.data)


    def scrollEvent(self, event):
        pass