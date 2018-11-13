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
from core import Experiment, Experiments

from GUI import View
from GUI import guiForFitOperation

import shelve

from IPython import embed

class Controller:

    def __init__(self):
        self.root = Tk.Tk()
        self.num_current_exp = 0
        self.model = Experiments.Experiments()
        self.current_exp = None
        self.list_of_exp = self.model.experiments
        self.current_measurement = None
        self.view = View.View(self.root, self)

        self.root.protocol("WM_DELETE_WINDOW",
                           self.on_quit)  # Exit when x pressed, notice that its the name of the function 'self.handler' and not a method call self.handler()

    def run(self):
        self.root.title("pySPC")
        self.root.deiconify()
        self.root.mainloop()

    ############

    def change_current_exp(self, num):
        if num == -1:
            num = len(self.list_of_exp) - 1
        self.num_current_exp = num
        self.current_exp = self.list_of_exp[num]
        self.update_all()

    def close_exp(self, num):
        del(self.list_of_exp[num])
        if num != 1:
            self.change_current_exp(num-1)

    def add_exp(self):
        self.model.add_new_exp()
        self.change_current_exp(-1)

    def open_SPC_File(self, file_path):
        # filePath = self.view.menu.askOpenFile('Choose the SPC file to analyse (.spc, .pt3, .ttt, ...')
        # TODO test extension
        exp = self.model.add_new_exp(file_path)
        self.current_exp = self.model.experiments[exp.file_name]
        self.current_exp.create_navigation_chronogram(0, 0, self.current_exp.data.channels[0].end_tick, self.current_exp.convert_seconds_in_ticks(self.current_exp.defaultBinSize_s))

        # FIXME le channel 0 is hardcoded
        self.view.currentTimeWindow = [0, self.current_exp.convert_ticks_in_seconds(self.current_exp.data.channels[0].end_tick) * 1E6]
        self.view.is_a_FileLoaded = True
        self.view.currentChannel = 0
        self.set_chrono_bin_size_s(0.01)

        # Put the file in the browser
        self.view.archi.status_area.insert_exp(self.current_exp)

        self.update_all()

    def generate_poisson_noise_file(self, time_s, count_per_secound):
        self.current_exp.new_exp("generate", ["Poisson", time_s, count_per_secound])

        # TODO don't duplicate from open SPC.
        # FIXME le channel 0 is hardcoded
        self.view.currentTimeWindow = [0, self.current_exp.convert_ticks_in_seconds(self.model.current_exp.channels[0].endTick) * 1E6]
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

        # self.view.archi.status_area.set_file_name(self.current_exp.file_name)
        # c = self.current_exp.data.channels[self.view.currentChannel]
        # self.view.archi.status_area.set_nb_of_photon_and_CPS(c.nb_of_tick, c.CPS)


    def get_analysis_start_end_tick(self):
        start_tick = 0
        end_tick = self.current_exp.data.channels[self.view.currentChannel].end_tick

        # Get time selection for data analysis
        selection_mode = self.view.archi.analyze_area.analyze_combo_box_source_sv.get()
        if selection_mode == "Whole":
            start_tick = 0
            end_tick = self.current_exp.data.channels[self.view.currentChannel].end_tick
        elif selection_mode == "Selection":
            start_tick = self.current_exp.convert_seconds_in_ticks(self.view.currentTimeWindow[0] / 1E6)
            end_tick = self.current_exp.convert_seconds_in_ticks(self.view.currentTimeWindow[1] / 1E6)
        elif selection_mode == "Time Zoom":
            start_tick = self.current_exp.convert_seconds_in_ticks(self.view.current_time_zoom_window[0] / 1E6)
            end_tick = self.current_exp.convert_seconds_in_ticks(self.view.current_time_zoom_window[1] / 1E6)

        if start_tick == end_tick:
            start_tick = 0
            end_tick = self.current_exp.data.channels[self.view.currentChannel].end_tick

        return start_tick, end_tick

    def display_measurement(self, measurement_name):
        measurement = self.current_exp.get_measurement(name)

        # Display data
        self.graph_measurement(measurement)

        # Display fit

    def calculate_measurement(self, measurement_name="current"):

        exp_name = self.current_exp.file_name
        if measurement_name == "current":
            measurement = self.current_measurement
        else:
            measurement = self.current_exp.get_measurement(measurement_name)
        param = None

        start_tick, end_tick = self.get_analysis_start_end_tick


        if measurement.type == "FCS":
            gui = self.view.archi.analyze_area.FCS_TimeAnalyze_gui
            num_c1 = int(gui.num_c1_sv.get())
            num_c2 = int(gui.num_c2_sv.get())
            max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
            param = [num_c1, num_c2, max_correlTime_ms]
        elif measurement.type == "lifetime":
            # TODO create Entry
            channel = 0
            self.current_exp.set_measurement_channel(measurement, channel)
            param = None

        self.view.archi.analyze_area.analyzePgb.start()
        self.model.calculate_measurement(exp_name, measurement.name, param)
        self.view.archi.analyze_area.analyzePgb.stop()
        self.view.archi.status_area.update_tree_view_line(measurement)

    def graph_measurement(self, measurement="current"):
        # t1_tick, t2_tick = self.get_analysis_start_end_tick()
        #
        # channel = self.view.currentChannel

        if measurement == "current":
            measurement = self.current_measurement

        guiGraphResult = self.view.archi.analyze_area.resultArea_gui.graph_results
        guiGraphResult.plot(measurement)


    def update_analyze(self):
        if self.view.is_a_FileLoaded is False:
            return

        t1_tick, t2_tick = self.get_analysis_start_end_tick()

        channel = self.view.currentChannel
        guiGraphResult = self.view.archi.analyze_area.resultArea_gui.graph_results

        # if self.view.currentOperation == "macro":
        #     pass
        #     # FIXME main channel and bin size
        #     # binSize_s = 100
        #     #
        #     # binInTick = self.current_exp.convert_seconds_in_ticks(binSize_s)
        #     # self.current_exp.results.chronograms[channel] = self.current_exp.chronogram(channel, t1_tick, t2_tick,
        #     #                                                                           binInTick)
        #     # # self.view.plotMainChronogram(self.model.results.chronograms[channel])
        #     # # TODO plot dans la fenetre result.
        #
        # elif self.view.currentOperation == "micro":
        #     #TODO name and comment
        #     lifetime = self.current_exp.micro_time_life_time(channel, t1_tick, t2_tick)
        #     guiGraphResult.plot("lifetime", lifetime)
        #
        # elif self.view.currentOperation == "FCS":
        #     gui = self.view.archi.analyze_area.FCS_TimeAnalyze_gui
        #     # FIXME recuperer les data
        #     channel1 = channel2 = channel
        #
        #     max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
        #     self.view.archi.analyze_area.analyzePgb.start()
        #     self.current_exp.FCS(channel1, channel2, t1_tick, t2_tick, max_correlTime_ms)
        #     self.view.archi.analyze_area.analyzePgb.stop()
        #     guiGraphResult.plot("FCS", self.current_exp.results.FCS_Measurements[channel])
        #
        # elif self.view.currentOperation == "DLS":
        #     gui = self.view.archi.analyze_area.DLS_TimeAnalyze_gui
        #     # FIXME recuperer les data
        #     channel1 = channel2 = channel
        #     max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
        #     start_time_mu_s = float(gui.start_time_micro_sv.get())
        #     precision = float(gui.corel_precision_sv.get())
        #     self.view.archi.analyze_area.analyzePgb.start()
        #     self.current_exp.DLS(channel1, channel2, t1_tick, t2_tick, max_correlTime_ms, start_time_mu_s, precision)
        #     self.view.archi.analyze_area.analyzePgb.stop()
        #     guiGraphResult.plot("DLS", self.current_exp.results.DLS_Measurements[channel])

    # def updateNavigation(self, channel, t1_microsec, t2_microsec, binSize_s=0.01):
    def update_navigation(self):
        if self.view.is_a_FileLoaded is False:
            return
        channel = self.view.currentChannel
        t1_microsec, t2_microsec = self.view.currentTimeWindow[0], self.view.currentTimeWindow[1]
        binSize_s = self.view.timezoom_bin_size_s

        # FIXME main channel and bin size
        t1_tick, t2_tick = self.current_exp.convert_seconds_in_ticks(t1_microsec / 1E6), self.current_exp.convert_seconds_in_ticks(
            t2_microsec / 1E6)
        bin_in_tick = self.current_exp.convert_seconds_in_ticks(binSize_s)

        self.current_exp.create_time_zoom_chronogram(channel, t1_tick, t2_tick, bin_in_tick)
        self.view.archi.navigation_area.timeZoom.graph_timeZoom.plot(self.current_exp.time_zoom_chronogram)

        self.view.archi.navigation_area.graph_navigation.plot(self.current_exp.navigation_chronogram,
                                                              self.view.currentTimeWindow[0],
                                                              self.view.currentTimeWindow[1])

        self.view.archi.navigation_area.timeZoom.chronoStart_sv.set(str(int(self.view.currentTimeWindow[0] / 1000)))
        self.view.archi.navigation_area.timeZoom.chronoEnd_sv.set(str(int(self.view.currentTimeWindow[1] / 1000)))

        self.current_exp.create_mini_PCH(channel)
        self.view.archi.navigation_area.timeZoom.graph_miniPCH.plot(self.current_exp.mini_PCH)

    def set_chrono_bin_size_s(self, binSize_s):
        self.view.archi.navigation_area.timeZoom.bin_size_micros_sv.set(str(binSize_s * 1E6))
        self.view.timezoom_bin_size_s = binSize_s

    # Measurement management
    def create_measurement(self, type, name, comment):
        start_tick, end_tick = self.get_analysis_start_end_tick()
        num_channel = self.view.currentChannel
        return self.current_exp.create_measurement(num_channel, start_tick, end_tick, type, name, comment)


    def add_measurement(self, measurement):
        self.current_exp.add_measurement(measurement)

    def del_measurement(self, name):
        self.current_exp.del_measurement(name)

    def duplicate_measurement(self, name):
        self.current_exp.duplicate_measurement(name)

    def set_current_measurement(self, name):
        if name in self.current_exp.measurements:
            measurement = self.current_exp.measurements[name]
            self.current_measurement = measurement
            return measurement
        else:
            return None
        # self.view.archi.analyze_area.set_current_measurement(measurement)

    def get_available_name_for_measurement(self, type):
        #FIXME
        if self.current_exp is None:
            return "no exp"
        else:
            return self.current_exp.get_available_name_for_measurement(type)

    def set_current_exp(self, exp_name):
        if exp_name in self.model.experiments:
            exp = self.model.experiments[exp_name]
            self.current_exp = exp
            return exp
        else:
            return None


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
            data = self.current_exp.results.lifetimes[channel]
            gui = self.view.archi.analyze_area.lifeTimeAnalyze_gui.guiForFitOperation_Lifetime
            fit_plot_mode = "lifetime"
        elif analyze_type == "DLS":
            data = self.current_exp.results.DLS_Measurements[channel]
            gui = self.view.archi.analyze_area.DLS_TimeAnalyze_gui.guiForFitOperation_DLS
            fit_plot_mode = "DLS"
        elif analyze_type == "FCS":
            data = self.current_exp.results.FCS_Measurements[channel]
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
        lf = self.current_exp.results.lifetimes[channel]
        lf.generate_artificial_IR(main_width, secondary_width, secondary_amplitude, time_offset)

    def export_graph_result(self, mode, file_path):
        self.view.archi.analyze_area.resultArea_gui.graph_results.export(mode, file_path)

    def save_state(self, savefile_path):
        self.shelf = shelve.open(savefile_path, 'n')  # n for new
        self.model.save_state(self.shelf)
        # self.current_exp.save_state(self.shelf)
        # self.view.saveState(self.shelf)
        self.shelf.close()

    def load_state(self, load_file_path):
        self.shelf = shelve.open(load_file_path)
        # self.current_exp.load_state(self.shelf)
        self.model.load_state(self.shelf)
        # self.view.loadState(self.shelf)
        self.shelf.close()

        self.update_all()
        # self.current_exp.update()
        # self.view.update()

    def log_message(self, msg):
        print(msg)
        self.view.archi.log_area.add_log_message(msg)

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