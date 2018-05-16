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

    def __init__(self, master_frame, view, controller, figsize, dpi):
        #super().__init__(masterFrame, view, controller, figsize, dpi)

        self.masterFrame = master_frame
        self.view = view
        self.controller = controller
        self.appearanceParam = view.appearenceParam

        self.x_selec_min = None
        self.x_selec_max = None

        self.x_selec_min_idx = None
        self.x_selec_max_idx = None

        self.ctrl_is_held = False
        self.shift_is_held = False



        self.type = None

        self.frame = tk.Frame(self.masterFrame)
        self.frame.pack(side="top", fill="both", expand=True)

        self.figure = plt.Figure(figsize=figsize, dpi=dpi)
        #[left, bottom, width, height]
        self.ax = self.figure.add_axes([0.08, 0.3, 0.9, 0.65], xticklabels=[])
        self.axResidual = self.figure.add_axes([0.08, 0.1, 0.9, 0.25])

        self.ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.createCallBacks()
        self.createWidgets()

    def plot(self, type_, data, is_plot_fit=False, is_zoom_x_selec=False, is_autoscale=False):
        self.type = type_
        self.data = data
        self.is_plot_fit = is_plot_fit
        self.ax.clear()
        self.axResidual.clear()

        self.data_y = y = data.data
        self.data_x = x = data.timeAxis
        self.data_fit = fit_y = data.eval_y_axis
        self.fit_x = fit_x = data.eval_x_axis
        self.data_residual = residuals_y = data.residuals

        # Default :
        plt_function = self.ax.plot
        plt_residual_fct = self.axResidual.plot

        self.ax.set_xlim(x[0], x[-1])
        self.axResidual.set_xlim(x[0], x[-1])

        if type_ == "lifetime":
            self.ax.set_xlabel("Microtime")
            self.ax.set_ylabel("Intensity")
            self.axResidual.set_xlabel("Microtime")
            self.axResidual.set_ylabel("Residual")

            # TODO : semilogy

            plt_function = self.ax.plot
            plt_residual_fct = self.axResidual.plot

        elif type_ == "FCS":
            self.ax.set_xlabel("time lag µs")
            self.ax.set_ylabel("Correlation")
            self.axResidual.set_xlabel("time lag µs")
            self.axResidual.set_ylabel("Residual")

            plt_function = self.ax.semilogx
            plt_residual_fct = self.axResidual.semilogx

        elif type_ == "DLS":
            self.ax.set_xlabel("time lag µs")
            self.ax.set_ylabel("Correlation")
            self.axResidual.set_xlabel("time lag µs")
            self.axResidual.set_ylabel("Residual")

            plt_function = self.ax.semilogx
            plt_residual_fct = self.axResidual.semilogx

        if self.x_selec_min is None :
            # There is no time selection on the graph
            plt_function(x, y)
            plt_residual_fct(x, np.zeros(np.size(y)))
            if is_plot_fit:
                self.ax.set_xlim(fit_x[0], fit_x[-1])
                plt_function(fit_x, fit_y, self.appearanceParam.line_type_fit_lifetime)
                plt_residual_fct(fit_x, residuals_y)
        else:

            x1 = np.searchsorted(x, self.x_selec_min)
            x2 = np.searchsorted(x, self.x_selec_max)

            self.x_selection_area = x[x1:x2]
            self.y_selection_area = y[x1:x2]

            if is_zoom_x_selec:
                self.ax.set_xlim(x[x1], x[x2])
                self.axResidual.set_xlim(x[x1], x[x2])

            plt_function(x[:x1], y[:x1], alpha=self.appearanceParam.alphaNonSelectedGraph)
            plt_function(x[x1:x2], y[x1:x2], alpha=1)
            plt_function(x[x2:], y[x2:], alpha=self.appearanceParam.alphaNonSelectedGraph)
            plt_residual_fct(x, np.zeros(np.size(y)))

            # Plot Fit
            if is_plot_fit:
                plt_function(fit_x, fit_y, self.appearanceParam.line_type_fit_lifetime)
                plt_residual_fct(fit_x, residuals_y)

        self.figure.canvas.draw()

    def replot(self, is_zoom_x_selec=False, is_autoscale=False):
        self.plot(self.type, self.data, self.is_plot_fit, is_zoom_x_selec, is_autoscale)

    def export(self, mode, file_path):

        if mode == "text":
            if self.data_fit is None:
                data = np.column_stack((self.data_x, self.data_y))
            else:
                data = np.column_stack((self.x_selection_area, self.y_selection_area, self.data_fit, self.data_residual))
            np.savetxt(file_path.name, data)
        elif mode == "script":
            f = open(file_path.name, "w")
            header = "import matplotlib.pyplot as plt" \
                     "import numpy as np"
            f.writelines(header)
            f.close()



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
        self.controller.set_lim_X_fit(self.x_selec_min, self.x_selec_max)
        self.plot(self.type, self.data)


    def scrollEvent(self, event):
        pass

