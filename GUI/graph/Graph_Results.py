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

        self.measurement = None

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

    def plot(self, measurement, is_plot_fit=False, is_zoom_x_selec=False, is_autoscale=False):
        if measurement is None:
            return

        self.type = measurement.type
        self.measurement = measurement
        self.is_plot_fit = is_plot_fit
        self.ax.clear()
        self.axResidual.clear()

        self.data_y = y = measurement.data
        self.data_x = x = measurement.time_axis
        self.data_fit = fit_y = measurement.eval_y_axis
        self.fit_x = fit_x = measurement.eval_x_axis
        self.data_residual = residuals_y = measurement.residuals

        if y is None:
            return

        # Default :
        plt_function = self.ax.plot
        plt_residual_fct = self.axResidual.plot

        self.ax.set_xlim(x[0], x[-1])
        self.axResidual.set_xlim(x[0], x[-1])

        if self.type == "lifetime":
            self.ax.set_xlabel("Microtime")
            self.ax.set_ylabel("Intensity")
            self.axResidual.set_xlabel("Microtime")
            self.axResidual.set_ylabel("Residual")

            plt_function = self.ax.semilogy
            plt_residual_fct = self.axResidual.plot

            if measurement.use_IR:
                rescale_factor = measurement.data.max() / measurement.IR_processed.max()
                self.ax.plot(measurement.IR_time_axis_processed, measurement.IR_processed*rescale_factor, "b--")


        elif self.type == "FCS":
            #TODO use canonic graph
            self.ax.set_xlabel("time lag µs")
            self.ax.set_ylabel("Correlation")
            self.axResidual.set_xlabel("time lag µs")
            self.axResidual.set_ylabel("Residual")

            plt_function = self.ax.semilogx
            plt_residual_fct = self.axResidual.semilogx

        elif self.type == "DLS":
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

                self.ax.set_ylim(np.min(self.y_selection_area), np.max(self.y_selection_area))
                if residuals_y is not None :
                    self.axResidual.set_ylim(np.min(residuals_y[x1:x2]), np.max(residuals_y[x1:x2]))

            if is_autoscale:
                self.ax.set_ylim(np.min(y), np.max(y))
                self.axResidual.set_ylim(np.min(residuals_y), np.max(residuals_y))

            plt_function(x[:x1], y[:x1], alpha=self.appearanceParam.alphaNonSelectedGraph)
            plt_function(x[x1:x2], y[x1:x2], alpha=1)
            plt_function(x[x2:], y[x2:], alpha=self.appearanceParam.alphaNonSelectedGraph)
            plt_residual_fct(x, np.zeros(np.size(y)))

            # Plot Fit
            if is_plot_fit:
                plt_function(fit_x, fit_y, self.appearanceParam.line_type_fit_lifetime)
                plt_residual_fct(fit_x, residuals_y)

        # selection patch
        if self.x_selec_min is not None :
            self.ax.add_patch(
                patches.Rectangle(
                    (self.x_selec_min, 0),  # (x,y)
                    self.x_selec_max-self.x_selec_min,  # width
                    measurement.data.max(),  # height
                    alpha=0.1
                )
            )

        self.figure.canvas.draw()

    def replot(self, is_zoom_x_selec=False, is_autoscale=False):
        self.plot(self.measurement, self.is_plot_fit, is_zoom_x_selec, is_autoscale)

    def zoom_to_x_selec(self):
        self.plot(self.measurement, self.is_plot_fit, is_zoom_x_selec=True, is_autoscale=False)

    def zoom_full(self):
        self.plot(self.measurement, self.is_plot_fit, is_zoom_x_selec=False, is_autoscale=False)

    def export(self, mode, file_path):

        if mode == "text":
            if self.data_fit is None:
                data = np.column_stack((self.data_x, self.data_y))
            else:
                #FIXME index ?

                export_size = min(self.x_selection_area.size, self.data_fit.size)

                data = np.column_stack((self.x_selection_area[0:export_size], self.y_selection_area[0:export_size], self.data_fit[0:export_size], self.data_residual[0:export_size]))
            np.savetxt(file_path.name, data, header="x data fit residual")
        elif mode == "script":
            f = open(file_path.name, "w")
            header = "import matplotlib.pyplot as plt" \
                     "import numpy as np"
            f.writelines(header)

            f.writelines("self.figure = plt.Figure(figsize=figsize, dpi=dpi")
            f.writelines("self.ax = self.figure.add_axes([0.08, 0.3, 0.9, 0.65], xticklabels=[])")
            f.writelines("self.axResidual = self.figure.add_axes([0.08, 0.1, 0.9, 0.25])")

            # self.ax.tick_params(
            #     axis='x',  # changes apply to the x-axis
            #     which='both',  # both major and minor ticks are affected
            #     bottom=False,  # ticks along the bottom edge are off
            #     top=False,  # ticks along the top edge are off
            #     labelbottom=False)  # labels along the bottom edge are off"

            f.close()
        elif mode == "image":
            self.figure.savefig(file_path)



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
        self.plot(self.measurement)


    def motion_notify_event(self, event):
        self.view.archi.analyze_area.resultArea_gui.set_xy_cursor_position(event.xdata, event.ydata)


