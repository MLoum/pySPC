#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note on MVC pattern :
# https://realpython.com/blog/python/the-model-view-controller-mvc-paradigm-summarized-with-legos/
# http://sametmax.com/quest-de-que-mvc-et-a-quoi-ca-sert/


# http://cirw.in//blog/time-to-move-on.html

# De toute façon, aucun MVC n’est parfait, et un peu de vue dégouline parfois sur le contrôleur,
# un peu de contrôleur coule dans le modèle, ou inversement. Il ne sert à rien d’être un nazi du MVC,
# c’est une bonne pratique, pas un dogme religieux.

import tkinter as Tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from core import ExpParam
from core import Results
from core import Data
from core import Experiment

from GUI import View
from GUI import guiForFitOperation

import shelve


class Controller:

    def __init__(self):
        self.root = Tk.Tk()
        self.list_of_exp = []
        self.num_current_exp = 0
        self.model = Experiment.Experiment()
        self.view = View.View(self.root, self)
        # Has to be done when all have been initialized
        self.view.archi.dock()
        # self.createMenuCommands()
        # self.createShortCut()
        self.root.protocol("WM_DELETE_WINDOW",
                           self.on_quit)  # Exit when x pressed, notice that its the name of the function 'self.handler' and not a method call self.handler()

        # self.view.sidepanel.plotBut.bind("<Button>",self.my_plot)
        # self.view.sidepanel.clearButton.bind("<Button>",self.clear)

    def run(self):
        self.root.title("pySPC")
        self.root.deiconify()
        self.root.mainloop()

    ############

    def change_current_exp(self, num):
        if num == -1:
            num = len(self.list_of_exp) - 1
        self.num_current_exp = num
        self.model = self.list_of_exp[num]
        self.update_all()

    def close_exp(self, num):
        del(self.list_of_exp[num])
        if num != 1:
            self.change_current_exp(num-1)

    def add_exp(self):
        self.list_of_exp.append(Experiment.Experiment())
        self.change_current_exp(-1)

    def open_SPC_File(self, file_path):
        # filePath = self.view.menu.askOpenFile('Choose the SPC file to analyse (.spc, .pt3, .ttt, ...')
        # TODO test extension
        self.model.new_exp("file", [file_path])

        # FIXME le channel 0 is hardcoded
        self.view.currentTimeWindow = [0, self.model.convert_ticks_in_seconds(self.model.data.channels[0].end_tick) * 1E6]
        self.view.is_a_FileLoaded = True
        self.view.currentChannel = 0
        self.set_chrono_bin_size_s(0.01)

        self.update_status()
        self.update_navigation()

        # self.view.archi.navigation_area.timeZoom.graph_timeZoom.plot(self.model.results.mainChronogram)
        # self.view.plotMainPCH(self.model.results.mainPCH)
        # self.view.archi.navigation_area.graph_navigation.plot(self.model.results.mainChronogram)

    def generate_poisson_noise_file(self, time_s, count_per_secound):
        self.model.new_exp("generate", ["Poisson", time_s, count_per_secound])

        # TODO don't duplicate from open SPC.
        # FIXME le channel 0 is hardcoded
        self.view.currentTimeWindow = [0, self.model.convert_ticks_in_seconds(self.model.data.channels[0].endTick) * 1E6]
        self.view.is_a_FileLoaded = True

        self.update_status()
        self.update_navigation()

    ############

    def update_all(self):
        self.update_status()
        self.update_navigation()
        self.update_analyze()

    def update_status(self):
        if not self.view.is_a_FileLoaded:
            return

        self.view.archi.status_area.setFileName(self.model.fileName)
        c = self.model.data.channels[self.view.currentChannel]
        self.view.archi.status_area.setNbOfPhotonAndCPS(c.nb_of_tick, c.CPS)

    def update_analyze(self):
        if self.view.is_a_FileLoaded is False:
            return

        t1_microsec = self.view.currentTimeWindow[0]
        t2_microsec = self.view.currentTimeWindow[1]
        t1_tick, t2_tick = self.model.convert_seconds_in_ticks(t1_microsec / 1E6), self.model.convert_seconds_in_ticks(
            t2_microsec / 1E6)
        channel = self.view.currentChannel
        guiGraphResult = self.view.archi.analyze_area.resultArea_gui.graph_results

        if self.view.currentOperation == "macro":
            # FIXME main channel and bin size
            binSize_s = 100

            binInTick = self.model.convert_seconds_in_ticks(binSize_s)
            self.model.results.chronograms[channel] = self.model.chronogram(channel, t1_tick, t2_tick,
                                                                                      binInTick)
            # self.view.plotMainChronogram(self.model.results.chronograms[channel])
            # TODO plot dans la fenetre result.

        elif self.view.currentOperation == "micro":
            self.model.micro_time_life_time(channel, t1_tick, t2_tick)
            guiGraphResult.plot("lifetime", self.model.results.lifetimes[channel])

        elif self.view.currentOperation == "FCS":
            gui = self.view.archi.analyze_area.FCS_TimeAnalyze_gui
            # FIXME recuperer les data
            channel1 = channel2 = channel

            max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
            self.view.archi.analyze_area.analyzePgb.start()
            self.model.FCS(channel1, channel2, t1_tick, t2_tick, max_correlTime_ms)
            self.view.archi.analyze_area.analyzePgb.stop()
            guiGraphResult.plot("FCS", self.model.results.FCS_Measurements[channel])

        elif self.view.currentOperation == "DLS":
            gui = self.view.archi.analyze_area.DLS_TimeAnalyze_gui
            # FIXME recuperer les data
            channel1 = channel2 = channel
            max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
            start_time_mu_s = float(gui.start_time_micro_sv.get())
            precision = float(gui.corel_precision_sv.get())
            self.view.archi.analyze_area.analyzePgb.start()
            self.model.DLS(channel1, channel2, t1_tick, t2_tick, max_correlTime_ms, start_time_mu_s, precision)
            self.view.archi.analyze_area.analyzePgb.stop()
            guiGraphResult.plot("DLS", self.model.results.DLS_Measurements[channel])

    # def updateNavigation(self, channel, t1_microsec, t2_microsec, binSize_s=0.01):
    def update_navigation(self):
        if self.view.is_a_FileLoaded is False:
            return
        channel = self.view.currentChannel
        t1_microsec, t2_microsec = self.view.currentTimeWindow[0], self.view.currentTimeWindow[1]
        binSize_s = 0.01

        # FIXME main channel and bin size
        t1_tick, t2_tick = self.model.convert_seconds_in_ticks(t1_microsec / 1E6), self.model.convert_seconds_in_ticks(
            t2_microsec / 1E6)
        binInTick = self.model.convert_seconds_in_ticks(binSize_s)
        self.model.results.chronograms[channel] = self.model.chronogram(channel, t1_tick, t2_tick, binInTick)
        self.view.archi.navigation_area.timeZoom.graph_timeZoom.plot(self.model.results.chronograms[channel])

        self.view.archi.navigation_area.graph_navigation.plot(self.model.results.navigationChronogram,
                                                              self.view.currentTimeWindow[0],
                                                              self.view.currentTimeWindow[1])

        self.view.archi.navigation_area.timeZoom.chronoStart_sv.set(str(int(self.view.currentTimeWindow[0] / 1000)))
        self.view.archi.navigation_area.timeZoom.chronoEnd_sv.set(str(int(self.view.currentTimeWindow[1] / 1000)))

    def set_chrono_bin_size_s(self, binSize_s):
        self.view.archi.navigation_area.timeZoom.binSizeMicros_sv.set(str(binSize_s * 1E6))
        self.view.currentBinSize_s = binSize_s

    def change_tab(self):
        if self.view.currentTab == 'macro':
            self.view.currentOperation = "macro"
            # self.update_analyze()

            # self.drawMainChronogram(self.view.currentChannel, self.view.currentTimeWindow[0], self.view.currentTimeWindow[1], self.view.currentBinSize_s)

        elif self.view.currentTab == 'micro':
            self.view.currentOperation = "micro"
            # self.update_analyze()

            # self.drawMicroTimeHisto(self.view.currentChannel, self.view.currentTimeWindow[0], self.view.currentTimeWindow[1])

    def create_short_cut(self):
        pass

    def set_lim_X_fit(self, idxStart, idxEnd):
        self.view.archi.analyze_area.FCS_TimeAnalyze_gui.guiForFitOperation_FCS.idx_lim_for_fit_min_sv.set(
            "%.2f"%(idxStart))
        self.view.archi.analyze_area.FCS_TimeAnalyze_gui.guiForFitOperation_FCS.idx_lim_for_fit_max_sv.set("%.2f"%(idxEnd))

        self.view.archi.analyze_area.lifeTimeAnalyze_gui.guiForFitOperation_Lifetime.idx_lim_for_fit_min_sv.set(
            "%.2f"%(idxStart))
        self.view.archi.analyze_area.lifeTimeAnalyze_gui.guiForFitOperation_Lifetime.idx_lim_for_fit_max_sv.set(
            "%.2f"%(idxEnd))

        self.view.archi.analyze_area.DLS_TimeAnalyze_gui.guiForFitOperation_DLS.idx_lim_for_fit_min_sv.set(
            "%.2f"%(idxStart))
        self.view.archi.analyze_area.DLS_TimeAnalyze_gui.guiForFitOperation_DLS.idx_lim_for_fit_max_sv.set("%.2f"%(idxEnd))

    def replot_result(self, is_zoom_x_selec=False, is_autoscale=False):
        self.view.archi.analyze_area.resultArea_gui.graph_results.replot(is_zoom_x_selec, is_autoscale)

    def fit(self, analyze_type, mode, model_name, params, idx_start=0, idx_end=-1):
        """
        :param analyze_type: is a string for lifetime FCS DLS ...
        :param mode can be fit, eval or guess
        :param model_name string for the model name
        :param params: list of paramaters for the fit
        xlims_idx : index of the limits of the x axis where to perform the fit.
        :return:

        """
        data, gui, fit_plot_mode = None, None, None

        channel = self.view.currentChannel
        # TODO cursor with fit limits.

        if analyze_type == "lifetime":
            data = self.model.results.lifetimes[channel]
            gui = self.view.archi.analyze_area.lifeTimeAnalyze_gui.guiForFitOperation_Lifetime
            fit_plot_mode = "lifetime"
        elif analyze_type == "DLS":
            data = self.model.results.DLS_Measurements[channel]
            gui = self.view.archi.analyze_area.DLS_TimeAnalyze_gui.guiForFitOperation_DLS
            fit_plot_mode = "DLS"
        elif analyze_type == "FCS":
            data = self.model.results.FCS_Measurements[channel]
            gui = self.view.archi.analyze_area.FCS_TimeAnalyze_gui.guiForFitOperation_FCS
            fit_plot_mode = "FCS"

        if data is not None:
            if mode == "eval":
                data.set_model(model_name)
                data.set_params(params)
                data.evalParams(idx_start, idx_end)

            elif mode == "guess":
                data.set_model(model_name)
                data.guess(idx_start, idx_end)
                gui.setParamsFromFit(data.params)

            elif mode == "fit":
                data.set_model(model_name)
                data.set_params(params)
                fitResults = data.fit(idx_start, idx_end)
                # TODO set option to tell if user want fit results exported to fit params
                gui.setParamsFromFit(data.params)
                self.view.archi.analyze_area.resultArea_gui.setTextResult(fitResults.fit_report())

            self.view.archi.analyze_area.resultArea_gui.graph_results.plot(fit_plot_mode, data, is_plot_fit=True)

    def shift_IR(self, main_width, secondary_width, secondary_amplitude, time_offset):
        channel = self.view.currentChannel
        lf = self.model.results.lifetimes[channel]
        lf.generateArtificialIR(main_width, secondary_width, secondary_amplitude, time_offset)

    def export_graph_result(self, mode, file_path):
        self.view.archi.analyze_area.resultArea_gui.graph_results.export(mode, file_path)

    def save_state(self, savefile_path):
        self.shelf = shelve.open(savefile_path, 'n')  # n for new

        self.model.save_state(self.shelf)
        self.view.saveState(self.shelf)
        self.shelf.close()

    def load_state(self, load_file_path):
        self.shelf = shelve.open(load_file_path)
        self.model.load_state(self.shelf)
        self.view.loadState(self.shelf)
        self.shelf.close()

        self.model.update()
        self.view.update()

    def on_quit(self):
        # paramFile = open('param.ini', 'w')
        # paramFile.write(self.saveDir)
        self.root.destroy()
        self.root.quit()

if __name__ == "__main__":


    #core =  model.PropSpecReaderModel()
    #ctrl = controller.controller(core)


    controller = Controller()
    controller.run()