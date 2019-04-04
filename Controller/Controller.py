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

from GUI import View, burst_analysis
from GUI import guiForFitOperation

import shelve

from IPython import embed

class Controller:

    def __init__(self):
        self.root = Tk.Tk()
        self.model = Experiments.Experiments()
        self.current_exp = None
        self.current_measurement = None
        self.view = View.View(self.root, self)

        self.root.protocol("WM_DELETE_WINDOW",
                           self.on_quit)  # Exit when x pressed, notice that its the name of the function 'self.handler' and not a method call self.handler()

    def run(self):
        self.root.title("pySPC")
        self.root.deiconify()
        self.root.deiconify()
        self.root.mainloop()

    ############

    def open_SPC_File(self, file_path):
        # filePath = self.view.menu.askOpenFile('Choose the SPC file to analyse (.spc, .pt3, .ttt, ...')
        # TODO test extension
        exp = self.model.add_new_exp("file", [file_path])
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
        exp = self.model.add_new_exp("generate", ["Poisson", time_s, count_per_secound])
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

    ############


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



    def update_all(self, is_full_update=False):
        self.update_status(is_full_update)
        self.update_navigation(is_full_update)
        self.update_analyze(is_full_update)

    def update_status(self, is_full_update=False):
        if not self.view.is_a_FileLoaded:
            return

        self.view.archi.status_area.update_tree_view()

        # self.view.archi.status_area.set_file_name(self.current_exp.file_name)
        # c = self.current_exp.data.channels[self.view.currentChannel]
        # self.view.archi.status_area.set_nb_of_photon_and_CPS(c.nb_of_tick, c.CPS)

    def update_analyze(self, is_full_update=False):
        if self.view.is_a_FileLoaded is False:
            return

        t1_tick, t2_tick = self.get_analysis_start_end_tick()

        channel = self.view.currentChannel
        guiGraphResult = self.view.archi.analyze_area.resultArea_gui.graph_results

        self.graph_measurement("current")


    # def updateNavigation(self, channel, t1_microsec, t2_microsec, binSize_s=0.01):
    def update_navigation(self, is_full_update=False):
        if self.view.is_a_FileLoaded is False:
            return

        #FIXME notion of "currentChannel"
        channel = self.view.currentChannel
        t1_microsec, t2_microsec = self.view.currentTimeWindow[0], self.view.currentTimeWindow[1]
        binSize_s = self.view.timezoom_bin_size_s

        # FIXME main channel and bin size
        t1_tick, t2_tick = self.current_exp.convert_seconds_in_ticks(t1_microsec / 1E6), self.current_exp.convert_seconds_in_ticks(
            t2_microsec / 1E6)
        bin_in_tick = self.current_exp.convert_seconds_in_ticks(binSize_s)

        # navigation
        if is_full_update :
            self.current_exp.create_navigation_chronogram(0, 0, self.current_exp.data.channels[0].end_tick, self.current_exp.convert_seconds_in_ticks(self.current_exp.defaultBinSize_s))
            self.view.currentTimeWindow = [0, self.current_exp.convert_ticks_in_seconds(
                self.current_exp.data.channels[0].end_tick) * 1E6]

        # Time zoom
        self.current_exp.create_time_zoom_chronogram(channel, t1_tick, t2_tick, bin_in_tick)
        self.view.archi.navigation_area.timeZoom.graph_timeZoom.plot(self.current_exp.time_zoom_chronogram)

        self.view.archi.navigation_area.graph_navigation.plot(self.current_exp.navigation_chronogram,
                                                              self.view.currentTimeWindow[0],
                                                              self.view.currentTimeWindow[1])

        self.view.archi.navigation_area.timeZoom.chronoStart_sv.set(str(int(self.view.currentTimeWindow[0] / 1000)))
        self.view.archi.navigation_area.timeZoom.chronoEnd_sv.set(str(int(self.view.currentTimeWindow[1] / 1000)))

        self.view.archi.navigation_area.filtre_t1_sv.set(str(self.view.current_time_zoom_window[0]))
        self.view.archi.navigation_area.filtre_t2_sv.set(str(self.view.current_time_zoom_window[1]))

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

    def get_measurement(self, measurement_name):
        return self.current_exp.get_measurement(measurement_name)

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

    def display_measurement(self, measurement_name):
        measurement = self.current_exp.get_measurement(measurement_name)

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

        measurement.start_tick, measurement.end_tick = self.get_analysis_start_end_tick()

        if measurement.type == "FCS":
            gui = self.view.archi.analyze_area.analyze_gui
            num_c1 = int(gui.num_c1_sv.get()) - 1
            num_c2 = int(gui.num_c2_sv.get()) - 1
            max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
            start_correlTime_ms = float(gui.startCorrelTime_sv.get())
            param = [num_c1, num_c2, start_correlTime_ms, max_correlTime_ms]
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

    def set_current_exp(self, exp_name):
        if exp_name in self.model.experiments:
            exp = self.model.experiments[exp_name]
            self.current_exp = exp
            return exp
        else:
            return None

    def get_experiment(self, exp_name):
        return self.model.get_exp(exp_name)

    def clear_exp(self):
        self.model = Experiments.Experiments()
        self.current_exp = None
        self.current_measurement = None
        self.view.archi.status_area.clear_treeview()


    def microtime_filter(self, num_channel, is_keep):
        #FIXME if absicce en ns
        micro_t1, micro_t2 = self.view.current_graph_result_limit
        micro_t1 = int(micro_t1 / self.current_exp.exp_param.mIcrotime_clickEquivalentIn_second*1E-9)
        micro_t2 = int(micro_t2 / self.current_exp.exp_param.mIcrotime_clickEquivalentIn_second*1E-9)
        self.current_exp.data.filter_micro_time(num_channel, micro_t1, micro_t2, is_keep, replacement_mode="nothing")
        self.update_all(is_full_update=True)

    def macrotime_time_selection_filter(self, t1_micro, t2_micro, num_channel, is_keep_time_selection, replacement_mode):
        t1_tick = self.current_exp.convert_seconds_in_ticks(t1_micro * 1E-6)
        t2_tick = self.current_exp.convert_seconds_in_ticks(t2_micro * 1E-6)
        self.current_exp.data.filter_time_selection(num_channel, t1_tick, t2_tick, is_keep_time_selection, replacement_mode)
        self.update_all(is_full_update=True)

    def macrotime_bin_threshold_filter(self, num_channel, threshold, is_keep, replacement_mode):
        bin_in_tick = self.current_exp.convert_seconds_in_ticks(self.view.timezoom_bin_size_s)
        self.current_exp.data.filter_bin_and_threshold(num_channel, threshold, bin_in_tick, is_keep, replacement_mode)
        self.update_all(is_full_update=True)

    def set_macrotime_filter_threshold(self, threshold):
        self.view.archi.navigation_area.timeZoom.graph_timeZoom.threshold = threshold
        self.view.archi.navigation_area.timeZoom.graph_miniPCH.threshold = threshold
        self.view.archi.navigation_area.filter_threshold_sv.set(str(threshold))
        self.update_navigation()

    def set_lim_X_fit(self, idx_start, idx_end):
        self.view.current_graph_result_limit = [idx_start, idx_end]

        if self.view.archi.analyze_area.gui_for_fit_operation is not None:
            gui = self.view.archi.analyze_area.gui_for_fit_operation
            gui.idx_lim_for_fit_min_sv.set("%.2f"%(idx_start))
            gui.idx_lim_for_fit_max_sv.set("%.2f" % (idx_end))

    def replot_result(self, is_zoom_x_selec=False, is_autoscale=False):
        self.view.archi.analyze_area.resultArea_gui.graph_results.replot(is_zoom_x_selec, is_autoscale)

    def guess_eval_fit(self, mode, model_name, params, idx_start=0, idx_end=-1):
        """
        :param mode can be fit, eval or guess
        :param model_name string for the model name
        :param params: list of paramaters for the fit
        xlims_idx : index of the limits of the x axis where to perform the fit.
        :return:

        """
        data, gui, fit_plot_mode = None, None, None

        measurement = self.current_measurement

        gui = self.view.archi.analyze_area.gui_for_fit_operation

        channel = self.view.currentChannel
        # TODO cursor with fit limits.

        measurement.set_model(model_name)

        if measurement.data is not None:
            if mode == "eval":
                measurement.set_params(params)
                measurement.eval(idx_start, idx_end)

            elif mode == "guess":
                measurement.guess(idx_start, idx_end)
                gui.setParamsFromFit(measurement.params)

            elif mode == "fit":
                measurement.set_params(params)
                fitResults = measurement.fit(idx_start, idx_end)
                # TODO set option to tell if user want fit results exported to fit params
                gui.setParamsFromFit(measurement.params)
                self.view.archi.analyze_area.resultArea_gui.setTextResult(fitResults.fit_report())

            self.view.archi.analyze_area.resultArea_gui.graph_results.plot(measurement, is_plot_fit=True)

    def shift_IR(self, main_width, secondary_width, secondary_amplitude, time_offset):
        channel = self.view.currentChannel
        lf = self.current_exp.results.lifetimes[channel]
        lf.generate_artificial_IR(main_width, secondary_width, secondary_amplitude, time_offset)

    def fit_IR(self, iniParams):
        fit_IR_results = self.current_measurement.fit_IR(iniParams)

        self.view.archi.analyze_area.resultArea_gui.setTextResult(fit_IR_results.fit_report())
        # FIXME
        self.view.archi.analyze_area.resultArea_gui.graph_results.plot(measurement, is_plot_fit=True)

    def export_graph_result(self, mode, file_path):
        self.view.archi.analyze_area.resultArea_gui.graph_results.export(mode, file_path)


    def launch_burst_analysis_GUI(self, name="", comment=""):
        burst_measurement = self.create_measurement("burst", name, comment)

        burst_analysis.BurstAnalysis_gui(self.root, self, self.view.appearenceParam, burst_measurement)

    def save_state(self, savefile_path):
        if self.current_exp.file_name is None:
            self.log_message("No experiment to save !\n")
            return
        self.shelf = shelve.open(savefile_path, 'n')  # n for new
        self.model.save_state(self.shelf)

        # save in the controller
        self.shelf['current_exo_name'] = self.current_exp.file_name
        if  self.current_measurement is None :
            self.shelf['current_measurement_name'] = "None"
        else:
            self.shelf['current_measurement_name'] = self.current_measurement.name

        self.view.saveState(self.shelf)

        self.shelf.close()

    def load_state(self, load_file_path):
        self.clear_exp()
        self.shelf = shelve.open(load_file_path)
        # self.current_exp.load_state(self.shelf)
        self.model.load_state(self.shelf)

        # load for the controller
        exp_name = self.shelf['current_exo_name']
        self.current_exp = self.get_experiment(exp_name)
        measurement_name = self.shelf['current_measurement_name']
        self.current_measurement = self.current_exp.get_measurement(measurement_name)


        self.view.loadState(self.shelf)

        self.shelf.close()

        self.view.is_a_FileLoaded = True

        #update
        self.view.archi.status_area.insert_list_of_exp(self.model.experiments)
        self.update_navigation(is_full_update=True)
        self.update_analyze(is_full_update=True)


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