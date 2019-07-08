import tkinter as tk

from tkinter import ttk

from tkinter import filedialog
#from pylab import *
import matplotlib.pyplot as plt
from IPython import embed
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# NavigationToolbar2TkAgg
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



class Graph_Results:
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
        self.alt_is_held = False

        self.measurement = None

        self.type = None

        self.zoom_factor_x = 1
        self.zoom_factor_y = 1
        self.shift_factor_x = 0
        self.shift_factor_y = 0

        self.is_plot_error_bar = True
        self.is_plot_all_FCS_curve = False
        self.is_autoscale = False
        self.is_zoom_x_selec = False
        self.is_plot_fit = True


        self.frame = tk.Frame(self.masterFrame)
        self.frame.pack(side="top", fill="both", expand=True)


        # self.figure = plt.Figure(figsize=figsize, dpi=dpi)
        # #[left, bottom, width, height]
        # self.ax = self.figure.add_axes([0.08, 0.3, 0.9, 0.65], xticklabels=[])
        # self.axResidual = self.figure.add_axes([0.08, 0.1, 0.9, 0.25])
        #
        # self.ax.tick_params(
        #     axis='x',  # changes apply to the x-axis
        #     which='both',  # both major and minor ticks are affected
        #     bottom=False,  # ticks along the bottom edge are off
        #     top=False,  # ticks along the top edge are off
        #     labelbottom=False)  # labels along the bottom edge are off


        self.figure, self.ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

        plt.subplots_adjust(hspace=0)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        # self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.canvas._tkcanvas.pack(side='top', fill='both', expand=1)

        self.createCallBacks()
        self.createWidgets()

    def plot(self, measurement):
        """
        Plot appearance param are hold in the view.plot_param object
        :param measurement:
        :return:
        """
        self.ax[0].clear()
        self.ax[1].clear()

        if measurement is None:
            self.figure.canvas.draw()
            return

        self.type = measurement.type
        self.measurement = measurement

        self.data_y = y = measurement.data
        self.data_x = x = measurement.time_axis
        self.error_bar = error_bar = measurement.error_bar
        self.data_fit = fit_y = measurement.eval_y_axis
        self.fit_x = fit_x = measurement.fit_x
        self.residual_x = residual_x = measurement.residual_x
        self.data_residual = residuals_y = measurement.residuals

        if y is None:
            return

        # default plot function
        plt_function = self.ax[0].plot
        plt_residual_fct = self.ax[1].plot
        plt_error_bar_fct = self.ax[0].plot

        self.ax[0].set_xlim(x[0], x[-1])
        self.ax[1].set_xlim(x[0], x[-1])

        if self.type == "lifetime":
            # self.ax[0].set_xlabel("Microtime")
            self.ax[0].set_ylabel("Intensity")
            self.ax[1].set_xlabel("Microtime")
            self.ax[1].set_ylabel("Residual")

            self.ax[0].tick_params(axis='both', which='major', labelsize=20)
            self.ax[0].tick_params(axis='both', which='minor', labelsize=8)
            self.ax[1].tick_params(axis='both', which='major', labelsize=20)
            self.ax[1].tick_params(axis='both', which='minor', labelsize=8)

            plt_function = self.ax[0].semilogy
            plt_residual_fct = self.ax[1].plot

            if measurement.use_IR:
                rescale_factor = measurement.data.max() / measurement.IRF.processed_data.max()
                self.ax[0].plot(measurement.IRF.time_axis_processed, measurement.IRF.processed_data*rescale_factor, "b--")
                if measurement.model is not None:
                    pass
                    # if measurement.model.non_convoluted_decay is not None:
                    #     self.ax[0].plot(measurement.time_axis, measurement.model.non_convoluted_decay, "r--", alpha=0.5)


        elif self.type == "FCS":
            #TODO use canonic graph
            self.ax[0].set_xlabel("time lag µs")
            self.ax[0].set_ylabel("Correlation")
            self.ax[1].set_xlabel("time lag µs")
            self.ax[1].set_ylabel("Residual")

            plt_function = self.ax[0].semilogx
            plt_residual_fct = self.ax[1].semilogx
            plt_error_bar_fct = self.ax[0].semilogx

        elif self.type == "DLS":
            self.ax[0].set_xlabel("time lag µs")
            self.ax[0].set_ylabel("Correlation")
            self.ax[1].set_xlabel("time lag µs")
            self.ax[1].set_ylabel("Residual")

            plt_function = self.ax[0].semilogx
            plt_residual_fct = self.ax[1].semilogx
            plt_error_bar_fct = self.ax[0].semilogx

        plt.subplots_adjust(hspace=0)

        for a in self.ax:
            a.grid(True)
            a.grid(True, which='minor', lw=0.3)

        if self.x_selec_min is None:
            # There is no time selection on the graph -> plot all time selection
            plt_function(x, y, self.appearanceParam.line_type_data, alpha=self.appearanceParam.alpha_data)
            plt_residual_fct(x, np.zeros(np.size(y)), self.appearanceParam.line_type_residual)
            if self.is_plot_fit and fit_y is not None:
                self.ax[0].set_xlim(fit_x[0], fit_x[-1])
                plt_function(fit_x, fit_y, self.appearanceParam.line_type_fit)
                plt_residual_fct(residual_x, residuals_y, self.appearanceParam.line_type_residual)
                ym = np.abs(residuals_y).max()
                self.ax[1].set_ylim(-ym, ym)
        else:
            # There is three areas for the plot :
            # - outside the time selection at the left
            # - inside the time selection
            # - outside the time selection at the right


            x1 = np.searchsorted(x, self.x_selec_min)
            x2 = np.searchsorted(x, self.x_selec_max)

            self.x_selection_area = x[x1:x2]
            self.y_selection_area = y[x1:x2]

            if self.is_zoom_x_selec:
                self.ax[0].set_xlim(x[x1], x[x2])
                self.ax[1].set_xlim(x[x1], x[x2])

                self.ax[0].set_ylim(np.min(self.y_selection_area), np.max(self.y_selection_area))
                if residuals_y is not None :
                    self.ax[1].set_ylim(np.min(residuals_y[x1:x2]), np.max(residuals_y[x1:x2]))

            elif self.is_autoscale:
                self.ax[0].set_ylim(np.min(y), np.max(y))
                self.ax[1].set_ylim(np.min(residuals_y), np.max(residuals_y))

            # outside the time selection at the left
            plt_function(x[:x1], y[:x1], self.appearanceParam.line_type_data_non_selected, alpha=self.appearanceParam.alpha_non_selected)
            # inside the time selection
            plt_function(x[x1:x2], y[x1:x2], self.appearanceParam.line_type_data, alpha=self.appearanceParam.alpha_selected)
            # outside the time selection at the right
            plt_function(x[x2:], y[x2:], self.appearanceParam.line_type_data_non_selected, alpha=self.appearanceParam.alpha_non_selected)

            plt_residual_fct(x, np.zeros(np.size(y)), self.appearanceParam.line_type_residual)

            # Plot Fit
            if self.is_plot_fit:
                # The fit data are usually inside the time selection area
                plt_function(fit_x, fit_y, self.appearanceParam.line_type_fit, label="fit")
                plt_residual_fct(residual_x, residuals_y, self.appearanceParam.line_type_residual)

        # Error Bar
        if self.is_plot_error_bar and error_bar is not None:
            plt_error_bar_fct(x, y + error_bar, alpha=self.appearanceParam.alpha_error_bar)
            plt_error_bar_fct(x, y - error_bar, alpha=self.appearanceParam.alpha_error_bar)

        if self.is_plot_all_FCS_curve and self.type == "FCS":
            if measurement.sub_correlation_curves is not None:
                NUM_COLORS = len(measurement.sub_correlation_curves)
                cm = plt.get_cmap('hot')
                for i, curve in enumerate(measurement.sub_correlation_curves):
                    plt_function(x, curve, self.appearanceParam.line_type_fit, color=cm(1. * i / NUM_COLORS), label="FCS_" + str(i))
                plt.legend()


        # selection patch
        if self.x_selec_min is not None:
            self.ax[0].add_patch(
                patches.Rectangle(
                    (self.x_selec_min, 0),  # (x,y)
                    self.x_selec_max-self.x_selec_min,  # width
                    measurement.data.max(),  # height
                    alpha=0.1
                )
            )

        self.figure.canvas.draw()

    def replot(self, is_zoom_x_selec=False, is_autoscale=False):
        self.plot(self.measurement)

    def zoom_to_x_selec(self):
        self.is_zoom_x_selec = True
        self.is_autoscale = False
        self.plot(self.measurement)

    def zoom_full(self):
        self.is_zoom_x_selec = False
        self.is_autoscale = False
        self.plot(self.measurement)

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
            #TODO
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
            self.figure.savefig(file_path, dpi=300)
            return True


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

    def scrollEvent(self, event):

        if event.button == 'up':
            if self.alt_is_held or self.controller.view.alt_is_held:
                # on y axis
                if self.ctrl_is_held or self.controller.view.ctrl_is_held:
                    #zoom
                    pass
                else:
                    # shift
                    self.shift_factor_y += self.appearanceParam.graph_result_shift_amount
            else:
                # on x ax
                if self.ctrl_is_held or self.controller.view.ctrl_is_held:
                    #zoom
                    pass
                else:
                    # shift
                    self.shift_factor_x += self.appearanceParam.graph_result_shift_amount


            # self.zoomOnMainChrono(self.currentTimeWindow[0], self.currentTimeWindow[1])
        elif event.button == 'down':
            if self.alt_is_held is True:
                # on y axis
                if self.ctrl_is_held is True:
                    #zoom
                    pass
                else:
                    # shift
                    self.shift_factor_y -= self.appearanceParam.graph_result_shift_amount
            else:
                # on x ax
                if self.ctrl_is_held is True:
                    #zoom
                    pass
                else:
                    # shift
                    self.shift_factor_x -= self.appearanceParam.graph_result_shift_amount
        self.replot()

    def createCallBacks(self):
        #https://stackoverflow.com/questions/18141069/matplotlib-how-to-pick-up-shift-click-on-figure
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.figure.canvas.mpl_connect('scroll_event', self.scrollEvent)
        self.figure.canvas.mpl_connect('button_press_event', self.button_press_event)
        self.figure.canvas.mpl_connect('button_release_event', self.button_release_event)
        self.figure.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)



    def createWidgets(self):
        # set useblit True on gtkagg for enhanced performance
        # TODO right button click ?
        # TODO Cursors
        self.spanSelec = SpanSelector(self.ax[0], self.onSpanSelect, 'horizontal', useblit=True,
                                    onmove_callback=self.onSpanMove,
                                    rectprops=dict(alpha=0.5, facecolor='red'), span_stays=True)


    def on_key_press(self, event):
        print(event.key)
        if event.key == 'ctrl':
            self.ctrl_is_held = True
        if event.key == 'alt':
            self.alt_is_held = True
        if event.key == 'shift':
            self.shift_is_held = True

    def on_key_release(self, event):
        if event.key == 'ctrl':
            self.ctrl_is_held = False
        if event.key == 'alt':
            self.alt_is_held = False
        if event.key == 'shift':
            self.shift_is_held = False

    # def button_press_event(self, event):
    #     #print('you pressed', event.button, event.xdata, event.ydata)
    #     if event.button == 2:
    #         pass
    #         # # click gauche
    #         # if event.key == "shift":
    #         # if self.measurement is not None:
    #         #     if self.measurement.type == "lifetime":
    #         #         if self.measurement.IR_raw is not None:
    #         #         # if event.dblclick:
    #         #             file_path = filedialog.asksaveasfile(title="Export file name ?",
    #         #                                                  initialdir=self.controller.view.saveDir)
    #         #             if file_path == None or file_path == '':
    #         #                 return None
    #         #             max_number_time = min(self.measurement.time_axis.size, self.measurement.IR_raw.size)
    #         #             max_number_time = min(max_number_time, self.measurement.data.size)
    #         #             data = np.column_stack((self.measurement.time_axis[0:max_number_time],
    #         #                                     self.measurement.IR_raw[0:max_number_time],
    #         #                                     self.measurement.data[0:max_number_time]))
    #         #             np.savetxt(file_path.name, data)
    #
    #
    #         #Test if filter mode or if the cursord is needed.
    #         #put horizontal cursor at mouse position
    #         # self.cursor_h.set_active(True)
    #         # self.cursor_h.eventson = True
    #         #activate cursor drag
    #         #tell other graph, via the controller ?, that the cursor has changed its position
    #         pass


    def button_press_event(self, event):
        pass

    def button_release_event(self, event):
        pass

