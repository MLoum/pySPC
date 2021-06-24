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
from core.analyze.lifetime import IRF
from core.liftetimeBenchmark.benchmark import LifeTimeBenchmark

from GUI import View, burst_analysis, LifetimeBench
from GUI import guiForFitOperation

import shelve



# from IPython import embed

class Controller:

    def __init__(self):
        self.root = Tk.Tk()
        self.model = Experiments.Experiments()
        self.current_exp = None
        self.current_measurement = None
        self.current_burst = None
        self.view = View.View(self.root, self)

        # alias
        self.graph_results = self.view.archi.analyze_area.resultArea_gui.graph_results
        # self.model.logger = self.view.logger

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
        param_dict = {}
        param_dict["file_path"] = file_path
        exp = self.model.add_new_exp("file", param_dict)
        self.current_exp = self.model.experiments[exp.file_name]
        self.current_exp.create_navigation_chronograms(0, self.current_exp.data.channels[0].end_tick, self.current_exp.convert_seconds_in_ticks(self.current_exp.defaultBinSize_s))

        self.current_measurement = None

        # FIXME le channel 0 is hardcoded
        self.view.currentTimeWindow = [0, self.current_exp.convert_ticks_in_seconds(self.current_exp.data.channels[0].end_tick) * 1E6]
        self.view.is_a_FileLoaded = True
        self.view.currentChannel = 0
        self.set_chrono_bin_size_s(0.01)


        # Put the file in the browser
        self.view.archi.status_area.insert_exp(self.current_exp)

        self.update_all()

    def generate_poisson_noise_file(self, params_dict):
        params_dict["type"] = "Poisson"
        exp = self.model.add_new_exp("generate", params_dict)
        self.current_exp = self.model.experiments[exp.file_name]
        self.current_exp.create_navigation_chronograms(0, self.current_exp.data.channels[0].end_tick, self.current_exp.convert_seconds_in_ticks(self.current_exp.defaultBinSize_s))

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
            #FIXME false after filtering
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
    def update_navigation(self, is_full_update=False, is_draw_burst=False, bursts=None):
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
        if is_full_update:
            self.current_exp.create_navigation_chronograms(0, self.current_exp.data.channels[0].end_tick, self.current_exp.convert_seconds_in_ticks(self.current_exp.defaultBinSize_s))
            self.view.currentTimeWindow = [0, self.current_exp.convert_ticks_in_seconds(
                self.current_exp.data.channels[0].end_tick) * 1E6]

        # Time zoom
        time_zoom_chronograms = self.current_exp.create_time_zoom_chronograms(t1_tick, t2_tick, bin_in_tick)

        # Overlay
        overlay = None
        if self.view.archi.analyze_area.analyze_gui is not None:
            type_ = self.view.archi.analyze_area.analyze_gui.type
            if type_ in ["lifetime", "FCS"]:
                if self.view.archi.analyze_area.analyze_gui.is_overlay_on_time_zoom.get():
                    x1, x2 = self.view.archi.analyze_area.gui_for_fit_operation.get_lim_for_fit()
                    if x1 != 0 and x2 != -1:
                        overlay = self.current_measurement.create_chronogram_overlay(time_zoom_chronograms, x1, x2)


        self.view.archi.navigation_area.time_zoom.graph_timeZoom.plot(self.current_exp.time_zoom_chronograms, overlay)

        if is_draw_burst:
            self.view.archi.navigation_area.graph_navigation.bursts = bursts
            #FIXME num channel is arbitrary set to zero
            self.view.archi.navigation_area.graph_navigation.plot(self.current_exp.navigation_chronograms,
                                                              self.view.currentTimeWindow[0],
                                                              self.view.currentTimeWindow[1], is_draw_burst=True)
        else:
            #FIXME num channel is arbitrary set to zero
            self.view.archi.navigation_area.graph_navigation.plot(self.current_exp.navigation_chronograms,
                                                              self.view.currentTimeWindow[0],
                                                              self.view.currentTimeWindow[1])

        self.view.archi.navigation_area.time_zoom.chronoStart_sv.set(str(int(self.view.currentTimeWindow[0] / 1000)))
        self.view.archi.navigation_area.time_zoom.chronoEnd_sv.set(str(int(self.view.currentTimeWindow[1] / 1000)))

        self.view.archi.navigation_area.filtre_t1_sv.set(str(self.view.current_time_zoom_window[0]))
        self.view.archi.navigation_area.filtre_t2_sv.set(str(self.view.current_time_zoom_window[1]))

        self.current_exp.create_mini_PCH()
        self.view.archi.navigation_area.time_zoom.graph_miniPCH.plot(self.current_exp.mini_PCHs)

    def set_chrono_bin_size_s(self, binSize_s):
        self.view.archi.navigation_area.time_zoom.bin_size_micros_sv.set(str(binSize_s * 1E6))
        self.view.time_zoom = binSize_s

    # Measurement management
    def create_measurement(self, type_, name, comment, additional_params=None):
        start_tick, end_tick = self.get_analysis_start_end_tick()
        num_channel = self.view.currentChannel

        # for key, value in additional_params.items():
        #     params[key] = value
        return self.current_exp.create_measurement(num_channel, start_tick, end_tick, type_, name, comment, is_store=True)


    def createBench(self, type_="lifetime", name="", comment="", additional_params=None):
        if type_=="lifetime":
            return LifeTimeBenchmark

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
        elif name is None:
            self.current_measurement = None
            return None
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
        name = self.current_exp.file_name + "_burst"
        measurement = self.current_exp.get_measurement(measurement_name)

        # Display data
        self.graph_measurement(measurement)

        # Display fit

    def calculate_measurement(self, measurement_name="current"):
        """
        Fetch the data on the GUI based on the type of the measurement and ask the core to calculate the measurement
        :param measurement_name:
        :return:
        """

        # Get the measurement
        exp_name = self.current_exp.file_name
        if measurement_name == "current":
            measurement = self.current_measurement
        else:
            measurement = self.current_exp.get_measurement(measurement_name)
        params = None

        # End and start of the measurement
        measurement.start_tick, measurement.end_tick = self.get_analysis_start_end_tick()

        gui = self.view.archi.analyze_area.analyze_gui
        if measurement.type == "FCS":
            is_multi_proc = gui.is_multiproc_iv.get()
            algo = gui.algo_combo_box_sv.get()
            num_c1 = int(gui.num_c1_sv.get()) - 1
            num_c2 = int(gui.num_c2_sv.get()) - 1
            max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
            start_correlTime_ms = float(gui.startCorrelTime_sv.get())
            precision = int(gui.precision_sv.get())
            params = [num_c1, num_c2, start_correlTime_ms, max_correlTime_ms, is_multi_proc, precision, algo]
        elif measurement.type == "lifetime":
            # TODO create Entry
            # FIXME multichannel
            channel = 0
            self.current_exp.set_measurement_channel(measurement, channel)
            params = None
        elif measurement.type == "phosphorescence":
            num_channel_start = int(gui.num_channel_start_sv.get()) - 1
            num_channel_stop = int(gui.num_channel_stop_sv.get()) - 1
            time_step_micros = float(gui.time_step_micros_sv.get())
            min_time_micros = float(gui.min_time_micros_sv.get())
            max_time_ms = float(gui.max_time_ms_sv.get())
            params = [num_channel_start, num_channel_stop, time_step_micros, min_time_micros, max_time_ms]

        elif measurement.type == "TOFPS":
            if gui.radio_micro_spec_sv.get() == "m":
                # Plot microtime
                channel = 0
                self.current_exp.set_measurement_channel(measurement, channel)
                params = None
            else:
                params = {}
                # plot spectrum
                params["wl_min"] = float(gui.wl_min_sv.get())
                params["wl_max"] = float(gui.wl_max_sv.get())
                params["fiber_length"] = float(gui.fiber_length_sv.get())
                params["microtime_calib"] = int(gui.microtime_calib_sv.get())
                params["wl_calib"] = float(gui.wl_calib_sv.get())

        self.view.archi.log_area.master_frame.focus_set()
        self.view.archi.log_area.logger.info("starting measurement calculation\n")
        self.view.archi.analyze_area.analyzePgb.start()

        self.model.calculate_measurement(exp_name, measurement.name, params)

        self.view.archi.analyze_area.analyzePgb.stop()
        self.view.archi.status_area.update_tree_view_line(measurement)
        self.view.archi.log_area.logger.info("calculaation complete\n")
        self.view.archi.analyze_area.master_frame.focus_set()


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
        self.view.archi.navigation_area.time_zoom.graph_timeZoom.threshold = threshold
        self.view.archi.navigation_area.time_zoom.graph_timeZoom.threshold_flank = None
        self.view.archi.navigation_area.time_zoom.graph_miniPCH.threshold = threshold

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

    def guess_eval_fit(self, mode, params, is_burst_analysis=False):
        """
        :param mode can be fit, eval or guess
        :param params : dict of parameters for the fit. See guiForFitOperation.get_fit_params
        :return:
        """
        data, gui, fit_plot_mode = None, None, None
        measurement = None
        if is_burst_analysis==False:
            measurement = self.current_measurement
            gui = self.view.archi.analyze_area.gui_for_fit_operation
        else:
            measurement = self.current_burst.measurement
            gui = self.view.burst_analysis_gui.analyze_gui.gui_for_fit_operation

        if measurement is None:
            return

        channel = self.view.currentChannel
        # TODO cursor with fit limits.
        model_name = params["model_name"]
        measurement.set_model(model_name)

        if measurement.data is not None:
            if mode == "eval":
                measurement.eval(params)

            elif mode == "guess":
                measurement.guess(params)
                gui.copy_param_from_fit(measurement.params)

            elif mode == "fit":
                fit_report = measurement.fit(params)
                # TODO set option to tell if user want fit results exported to fit params
                # gui.setParamsFromFit(measurement.params)
                # self.view.archi.analyze_area.resultArea_gui.setTextResult(fitResults.fit_report())
                self.view.archi.analyze_area.resultArea_gui.setTextResult(fit_report)

            self.view.graph_result.is_plot_fit = True
            self.view.graph_result.plot(measurement)

    def set_use_error_bar(self, is_error_bar_for_fit):
        self.current_measurement.is_error_bar_for_fit = bool(is_error_bar_for_fit)

    #IRF
    def open_and_set_IRF_file(self, file_path):
        irf = IRF(self.model, self.current_exp, file_path)
        self.model.add_irf(irf)
        self.current_measurement.IRF = irf
        return irf.name

    def generate_IRF(self, params_dict):
        #TODO merge with open and set IRF
        irf = IRF(self.model, self.current_exp)
        #FIXME algo et ajout dans param_dict
        params_dict["irf_shift"] = 0
        irf.generate(params_dict, algo="Becker")
        self.model.add_irf(irf)
        self.current_measurement.IRF = irf
        return irf.name

    def set_IRF(self, irf_name):
        self.current_measurement.IRF = self.model.irf[irf_name]


    # def shift_IR(self, main_width, secondary_width, secondary_amplitude, time_offset):
    #     channel = self.view.currentChannel
    #     lf = self.current_exp.results.lifetimes[channel]
    #     lf.generate_artificial_IR(main_width, secondary_width, secondary_amplitude, time_offset)

    def fit_IR(self, iniParams):
        fit_IR_results = self.current_measurement.fit_IR(iniParams)

        self.view.archi.analyze_area.resultArea_gui.setTextResult(fit_IR_results.fit_report())
        # FIXME
        self.view.archi.analyze_area.resultArea_gui.graph_results.plot(measurement, is_plot_fit=True)



    def export_graph_result(self, mode, file_path):
        result = False
        result = self.view.archi.analyze_area.resultArea_gui.graph_results.export(mode, file_path)
        if result:
            self.log_message("Image from graph result saved to " + file_path + "\n")

    def launch_burst_analysis_GUI(self, name="", comment=""):
        burst_measurement = self.create_measurement("burst", name, comment)

        self.view.burst_analysis_gui = burst_analysis.BurstAnalysis_gui(self.root, self, self.view.appearenceParam, burst_measurement)


    def launch_bench_LF_GUI(self):
        self.life_time_benchmark = LifeTimeBenchmark()
        self.view.bench_LF_fitting = LifetimeBench.LifetimeBench_gui(self.root, self, self.view.appearenceParam, self.life_time_benchmark)


    def launch_bench(self, params):
        #TODO new thread and feedback from algorithm
        if self.life_time_benchmark is not None:
            self.life_time_benchmark.create_data(params)


    def validate_burst_selection(self, bursts_measurement):
        self.add_measurement(bursts_measurement)
        self.view.archi.status_area.insert_measurement(bursts_measurement)

    def display_burst(self, burst, measurement):
        # Display the burst in the timezoom windows
        self.view.timezoom_bin_size_s = self.current_exp.convert_ticks_in_seconds(measurement.bin_in_tick)
        #in µS
        # We want to display the burst with its onset and downset. A third of the windows is dedicated for the burst, a third for the onset (i.e the noise before the burst) and 1/3 for the downset
        burst_start_micro_s = self.current_exp.convert_ticks_in_seconds(burst.tick_start)*1E6
        burst_end_micro_s = self.current_exp.convert_ticks_in_seconds(burst.tick_end) * 1E6
        burst_duration_micro = burst_end_micro_s - burst_start_micro_s

        burst_start_micro_s = burst.num_bin_start * self.current_exp.convert_ticks_in_seconds(measurement.bin_in_tick)*1E6
        burst_end_micro_s = burst.num_bin_end * self.current_exp.convert_ticks_in_seconds(measurement.bin_in_tick)*1E6
        burst_duration_micro = burst_end_micro_s - burst_start_micro_s

        # Wa want at least 50 (?) bins on screen

        min_nb_of_bin_to_display = 50
        coeff_visualization = 1
        if 3*burst.nb_bin < min_nb_of_bin_to_display:
            coeff_visualization = ((min_nb_of_bin_to_display - burst.nb_bin)/2)/ burst.nb_bin


        self.view.currentTimeWindow[0] = max(0, burst_start_micro_s - coeff_visualization*burst_duration_micro)
        last_tick_micro = self.current_exp.convert_ticks_in_seconds(self.current_exp.data.channels[0].end_tick)*1E6
        self.view.currentTimeWindow[1] = min(last_tick_micro, burst_end_micro_s + coeff_visualization*burst_duration_micro)

        self.view.current_time_zoom_window[0] = burst_start_micro_s
        self.view.current_time_zoom_window[1] = burst_end_micro_s

        self.view.archi.navigation_area.time_zoom.graph_timeZoom.threshold = measurement.burst_threshold
        self.view.archi.navigation_area.time_zoom.graph_timeZoom.threshold_flank = measurement.flank_threshold
        self.view.archi.navigation_area.time_zoom.graph_miniPCH.threshold = measurement.burst_threshold

        self.view.archi.navigation_area.filter_threshold_sv.set(str(measurement.threshold))


        # Set the ylim to the maximum of the burst.
        self.update_navigation(is_draw_burst=True, bursts=measurement.bursts)

        # Graph the eventual measurement of the burst
        self.graph_measurement(burst.measurement)
        pass

    def launch_burst_measurement(self, burst_measurement, type="lifetime", model_name=None, fit_params=None, idx_start=0, idx_end=-1):
        burst_measurement.perform_measurements(self, type, model_name, fit_params, idx_start, idx_end)

    def save_state(self, savefile_path):
        if self.current_exp is None:
            self.log_message("No experiment to save !\n")
            return
        self.shelf = shelve.open(savefile_path, 'n')  # n for new
        self.model.save_state(self.shelf)

        # save in the controller
        self.shelf['current_exo_name'] = self.current_exp.file_name
        if self.current_measurement is None :
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


    # Debug tools

    def get_raw_data(self):
        #FIXME channel
        #self.model.get_raw_data(self.current_exp, num_channel=0)
        self.model.export_raw_data(self.current_exp, num_channel=0)

if __name__ == "__main__":


    #core =  model.PropSpecReaderModel()
    #ctrl = controller.controller(core)


    controller = Controller()
    controller.run()