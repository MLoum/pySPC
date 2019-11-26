import tkinter as tk
from tkinter import ttk


#from pylab import *
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

import numpy as np


from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Cursor
from matplotlib.widgets import MultiCursor

from .interactiveGraphs import InteractiveGraph

class Graph_navigation(InteractiveGraph):
    """
    Il s'agit d'un graphe montrant la totalité du fichier.
    On peut cliquer et selctionner des zones pour naviguer dans le fichier
    """

    def __init__(self, master_frame, view, controller, figsize, dpi):
        super().__init__(master_frame, view, controller, figsize, dpi)
        self.ax.axis('off')
        self.main_chronos = None
        self.bursts = None
        self.figure.tight_layout()
        self.createCallBacks()
        self.createWidgets()

    #TODO share common code with graph time zoom ?
    def plot(self, main_chronos, tSelec_1=0, tSelec_2=-1, is_draw_burst=None):
        self.main_chronos = main_chronos
        self.ax.clear()

        max_data = -1
        for num_channel in self.view.displayed_channel:
            chrono = main_chronos[num_channel]
        #reduce nb of point to something aroudn 1000 (approximative size in pixel
            if chrono.nb_of_bin > self.view.appearenceParam.max_pixel_for_chrono:
                skipsize = int(chrono.nb_of_bin / self.view.appearenceParam.max_pixel_for_chrono)
                idx = np.arange(0, len(chrono.data), skipsize)
                chrono_y = chrono.data[idx]
                chrono_plot_x = chrono.time_axis[idx]
            else:
                chrono_y = chrono.data
                chrono_plot_x = chrono.time_axis

            if max(chrono_y) > max_data:
                max_data = max(chrono_y)

            # self.ax.plot(plotX, plot)
            self.ax.plot(chrono_plot_x, chrono_y, self.view.appearenceParam.channels_trace_color[num_channel])


        # Autoscale
        max_time = -1
        min_time = 1E35
        for num_channel in self.view.displayed_channel:
            if main_chronos[num_channel].time_axis[-1] > max_time:
                max_time = main_chronos[num_channel].time_axis[-1]
            if main_chronos[num_channel].time_axis[0] < min_time:
                min_time = main_chronos[num_channel].time_axis[0]

        self.ax.set_xlim(min_time, max_time)

        if is_draw_burst and self.bursts is not None:
            x_lines = []
            for burst in self.bursts:
                burst_mean_timestamp = (burst.tick_end + burst.tick_start)/2
                burst_micros = self.controller.current_exp.convert_ticks_in_seconds(burst_mean_timestamp)*1E6
                x_lines.append(burst_micros)
            #FIXME 0 for min position of the lines ?
            self.ax.vlines(x_lines, 0, max_data)


        # Selection Patch
        x_start = tSelec_1
        if tSelec_2 == -1:
            x_end = main_chronos.time_axis.max()
        else:
            x_end = tSelec_2

        self.ax.add_patch(
            patches.Rectangle(
                (x_start, 0),  # (x,y)
                x_end-x_start,  # width
                max_data,  # height
                alpha=0.2
            )
        )

        self.figure.canvas.draw()


    def onSpanMove(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        self.view.currentTimeWindow = [xmin, xmax]
        self.controller.update_navigation()
        #self.controller.zoom()

    def onSpanSelect(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        self.view.currentTimeWindow = [xmin, xmax]
        self.controller.update_navigation()

    def scrollEvent(self, event):
        if self.ctrl_is_held==True:
            #On zoom
            if event.button == 'up':
                mean = (self.view.currentTimeWindow[0] + self.view.currentTimeWindow[1]) / 2
                delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
                self.view.currentTimeWindow[0] = self.view.currentTimeWindow[0] + delta / 4
                self.view.currentTimeWindow[1] = self.view.currentTimeWindow[1] - delta / 4

                # self.zoomOnMainChrono(self.currentTimeWindow[0], self.currentTimeWindow[1])
            elif event.button == 'down':
                mean = (self.view.currentTimeWindow[0] + self.view.currentTimeWindow[1]) / 2
                delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
                self.view.currentTimeWindow[0] = mean - delta * 2
                self.view.currentTimeWindow[1] = mean + delta * 2
                self.view.currentTimeWindow[0] = max(0, self.view.currentTimeWindow[0])
                # self.currentTimeWindow[1] = min(self.currentTimeWindow[1], )

            self.controller.update_navigation()
        else:
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

            self.controller.update_navigation()

