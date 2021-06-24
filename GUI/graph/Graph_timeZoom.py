import tkinter as tk
from tkinter import ttk


#from pylab import *
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import numpy as np

from tkinter import filedialog

from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Cursor
from matplotlib.widgets import MultiCursor
import matplotlib.patches as patches

from .interactiveGraphs import InteractiveGraph

# class Graph_timeZoom(InteractiveGraph):
class Graph_timeZoom():
    """
        Il s'agit d'une zone où on peut zoomer sur la time trace pour faire des analyses précises,
        voire à la particule unique si le S/N le permet.

        On aussi sur la droite un Photon Counting Histogramm qui est bien pratique pour fixer des seuils
        de filtrage.

        Il y a aussi une zone permettant d'identifier certaine partie de la courbe de temps de vie ou de corrélation.
        :return:
    """

    def __init__(self, master_frame, view, controller, figsize, dpi):
        # super().__init__(master_frame, view, controller, figsize, dpi)
        #self.ax.axis('off')


        self.masterFrame = master_frame
        self.view = view
        self.controller = controller
        self.appearanceParam = view.appearenceParam

        self.ctrl_is_held = False
        self.shift_is_held = False
        self.alt_is_held = False

        self.data_x = None
        self.data_y = None
        self.data_fit = None
        self.data_residual = None

        self.frame = tk.Frame(self.masterFrame)
        self.frame.pack(side="top", fill="both", expand=True)

        # self.figure = plt.Figure(figsize=figsize, dpi=dpi)
        # self.ax = self.figure.add_subplot(111)
        #
        # self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        # self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        # self.figure, self.ax = plt.subplots(2, 1, figsize=(18, 8), dpi=50, sharex=True,
        #                        gridspec_kw={'height_ratios': [9, 1]})

        self.figure = plt.Figure(figsize=(18,8), dpi=100)
        self.ax = self.figure.add_subplot(111)

        plt.subplots_adjust(hspace=0)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        # self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.canvas._tkcanvas.pack(side='top', fill='both', expand=1)

        self.threshold = None
        self.threshold_flank = None
        self.chronos = None


        self.createWidgets()
        self.createCallBacks()

    def plot(self, chronos, overlay=None):
        self.chronos = chronos

        self.ax.clear()

        if not self.view.displayed_channel:
            return

        max_data = 0
        for num_channel in self.view.displayed_channel:
            chrono = chronos[num_channel]
        #reduce nb of point to something aroudn 1000 (approximative size in pixel
            if chrono.nb_of_bin > self.view.appearenceParam.max_pixel_for_chrono:
                skipsize = int(chrono.nb_of_bin / self.view.appearenceParam.max_pixel_for_chrono)
                idx = np.arange(0, len(chrono.data), skipsize)

                chrono_y = chrono.data[idx]
                chrono_plot_x = chrono.time_axis[idx]

                if overlay is not None:
                    #FIXME overlay broken with multichannel
                    overlay = overlay[idx]


            else:
                chrono_y = chrono.data
                chrono_plot_x = chrono.time_axis

            if max(chrono_y) > max_data:
                max_data = max(chrono_y)

            self.ax.plot(chrono_plot_x, chrono_y, self.view.appearenceParam.channels_trace_color[num_channel])
            self.ax.fill_between(chrono_plot_x, 0, chrono_y,
                                 alpha=self.view.appearenceParam.channels_trace_alpha[num_channel])
            # self.ax.bar(chrono_plot_x, chrono_y, width=0.8, color=self.view.appearenceParam.channels_trace_color[num_channel], linewidth=5)


        # TODO fix this pythonic one line tentative
        # self.ax.set_xlim(min(chronos, key=lambda x: x.time_axis[-1]), max(chronos, key=lambda x: x.time_axis[-1]))

        max_time = -1
        min_time = 1E35
        for num_channel in self.view.displayed_channel:
            if chronos[num_channel].time_axis[-1] > max_time:
                max_time = chronos[num_channel].time_axis[-1]
            if chronos[num_channel].time_axis[0] < min_time:
                min_time = chronos[num_channel].time_axis[0]

        self.ax.set_xlim(min_time, max_time)

        #FIXME appareance param ?
        self.ax.set_xlabel("time / µs", fontsize=20)
        self.ax.set_ylabel("Intensity", fontsize=20)

        self.ax.tick_params(axis='both', which='major', labelsize=20)
        self.ax.tick_params(axis='both', which='minor', labelsize=8)

        if self.threshold is not None:
            #FIXME nb channel assumed to be 0
            chrono = chronos[0]
            self.ax.hlines(self.threshold, chrono.time_axis[0], chrono.time_axis.max(), linewidth=4)

        if self.threshold_flank is not None:
            #FIXME nb channel assumed to be 0
            chrono = chronos[0]
            self.ax.hlines(self.threshold_flank, chrono.time_axis[0], chrono.time_axis.max(), linewidth=4)

        if self.view.current_time_zoom_window != [0, 0]:
            self.ax.add_patch(
                patches.Rectangle(
                    (self.view.current_time_zoom_window[0], 0),  # (x,y)
                    self.view.current_time_zoom_window[1]-self.view.current_time_zoom_window[0],  # width
                    max_data,  # height
                    alpha=0.2   #TODO appareance param
                )
            )

        # FIXME Overlay
        if overlay is not None:
            # Overlay is a 1D array
            nb_y_point = 10
            # FIXME multi channel
            y = np.linspace(0, chronos[0].data.max(), nb_y_point)
            # z = np.repeat(overlay, nb_y_point, axis=0)
            z = np.tile(overlay, (nb_y_point, 1))
            self.ax.contourf(chrono_plot_x, y, z, 30, alpha=0.3, cmap=plt.cm.hot)


        self.figure.canvas.draw()

    def scrollEvent(self, event):
        if self.shift_is_held==False:
            #On translate
            if event.button == 'up':
                delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
                self.view.currentTimeWindow[0] += delta / 2
                self.view.currentTimeWindow[1] += delta / 2

                # TODO vérfier borne depassement fichier.
            elif event.button == 'down':
                delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
                self.view.currentTimeWindow[0] -= delta / 2
                self.view.currentTimeWindow[1] -= delta / 2
                if self.view.currentTimeWindow[0] < 0:
                    self.view.currentTimeWindow[0] = 0

            #FIXME something is wrong here and make the soft carchshs
            # self.controller.zoom()
        else:
            #On zoom
            pass


    def button_press_event(self, event):
        #print('you pressed', event.button, event.xdata, event.ydata)
        if event.button == 2:
            # pass
            # # click gauche
            # if event.key == "shift":

            # if event.dblclick:
            file_path = filedialog.asksaveasfile(title="Export file name ?",
                                                 initialdir=self.controller.view.saveDir)
            if file_path == None or file_path == '':
                return None

            # Export the graph
            if self.chronos is not None:
                data = np.column_stack((self.chronos.time_axis,
                                        self.chronos.data))
                np.savetxt(file_path.name, data, header="time chrono")

            #Test if filter mode or if the cursord is needed.
            #put horizontal cursor at mouse position
            # self.cursor_h.set_active(True)
            # self.cursor_h.eventson = True
            #activate cursor drag
            #tell other graph, via the controller ?, that the cursor has changed its position
            pass

    def createCallBacks(self):
        #https://stackoverflow.com/questions/18141069/matplotlib-how-to-pick-up-shift-click-on-figure
        # self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        # self.figure.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.figure.canvas.mpl_connect('scroll_event', self.scrollEvent)
        self.figure.canvas.mpl_connect('button_press_event', self.button_press_event)
        self.figure.canvas.mpl_connect('button_release_event', self.button_release_event)
        self.figure.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)

    def button_release_event(self, event):
        if event.button == 1:
            pass
            # click gauche

            #self.cursor_h.set_active(False)
            #self.cursor_h.eventson = False
            # self.cursor_h.visible = True
            #self.cursor_h.drawon = True

    def onSpanMove(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        self.view.current_time_zoom_window = [xmin, xmax]
        # self.controller.zoom()

    def onSpanSelect(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        self.view.current_time_zoom_window = [xmin, xmax]
        self.controller.update_navigation()

    def createWidgets(self):
        # set useblit True on gtkagg for enhanced performance
        # TODO right button click ?
        # TODO Cursors
        self.spanSelec = SpanSelector(self.ax, self.onSpanSelect, 'horizontal', useblit=True,
                                    onmove_callback=self.onSpanMove,
                                    rectprops=dict(alpha=0.5, facecolor='red'), span_stays=True)
        # self.cursor_h = Cursor(self.ax, useblit=True, color='red', horizOn=True, vertOn=False, linewidth=2)
        #
        # self.cursor_h.set_active(False)
        # self.cursor_h.drawon = True
        #drawon
        #eventson
        # self.setOnOffCursors(True)


    def motion_notify_event(self, event):
        self.view.archi.navigation_area.time_zoom.set_xy_cursor_position(event.xdata, event.ydata)

    def cursorMove(self, event):
        print('Cursor move')
        print(event)


