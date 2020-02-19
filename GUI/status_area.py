import tkinter as tk
from tkinter import ttk
# from IPython import embed


class Status_area():
    def __init__(self, master_frame, view, controller, appearence_param):
        self.master_frame = master_frame
        self.view = view
        self.controller = controller
        self.appearence_param = appearence_param
        self.exp_iid_dict = {}
        self.mes_iid_dict = {}

    def populate(self):
        # TreeView

        self.frame_tree_view = tk.LabelFrame(self.master_frame, text="File and measurement browser", borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_tree_view.pack(side="top", fill="both", expand=True)

        #https://riptutorial.com/tkinter/example/31885/customize-a-treeview
        self.tree_view = ttk.Treeview(self.frame_tree_view)
        self.tree_view["columns"] = ("exp name", "burst num", "meas. name", "type", "nb photon", "CPS", "channel", "comment", "t_start_µs", "t_end_µs")
        # remove first empty column with the identifier
        # self.tree_view['show'] = 'headings'
        # tree.column("#0", width=270, minwidth=270, stretch=tk.NO) tree.column("one", width=150, minwidth=150, stretch=tk.NO) tree.column("two", width=400, minwidth=200) tree.column("three", width=80, minwidth=50, stretch=tk.NO)
        self.tree_view.column("#0", width=25, stretch=tk.NO)
        self.tree_view.column("exp name", width=300, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("burst num", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("meas. name", width=300, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("type", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("nb photon", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("CPS", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("channel", width=75, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("comment", width=600, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("t_start_µs", width=50, stretch=tk.YES, anchor=tk.CENTER)
        self.tree_view.column("t_end_µs", width=50, stretch=tk.YES, anchor=tk.CENTER)


        self.tree_view.heading("exp name", text="Exp name")
        self.tree_view.heading("burst num", text="burst num")
        self.tree_view.heading("meas. name", text="Meas. name")
        self.tree_view.heading("type", text="Type")
        self.tree_view.heading("nb photon", text="Nb photon")
        self.tree_view.heading("CPS", text="CPS")
        self.tree_view.heading("channel", text="channel")
        self.tree_view.heading("comment", text="Comment")
        self.tree_view.heading("t_start_µs", text="t_start_µs")
        self.tree_view.heading("t_end_µs", text="t_end_µs")

        ysb = ttk.Scrollbar(self.frame_tree_view, orient='vertical', command=self.tree_view.yview)
        self.tree_view.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        self.tree_view.configure(yscroll=ysb.set)

        self.tree_view.bind('<<TreeviewSelect>>', self.treeview_measurement_select)

        self.tree_view.grid(row=0, column=0)

        self.frame_cmd = tk.Frame(self.frame_tree_view)
        self.frame_cmd.grid(row=1, column=0)

        self.frame_tree_view_cmd_exp = tk.LabelFrame(self.frame_cmd, text="Exp File",
                      borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_tree_view_cmd_exp.grid(row=1, column=0)

        ttk.Button(self.frame_tree_view_cmd_exp, text="+", width=10, command=self.add_exp_file).grid(row=0, column=0)
        ttk.Button(self.frame_tree_view_cmd_exp, text="-", width=10, command=self.del_exp_file).grid(row=0, column=1)
        ttk.Button(self.frame_tree_view_cmd_exp, text="Duplicate", width=10, command=self.duplicate_exp_fil).grid(row=0, column=2)


        self.frame_tree_view_cmd_mes = tk.LabelFrame(self.frame_cmd, text="Measurement",
                      borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_tree_view_cmd_mes.grid(row=1, column=1)

        ttk.Button(self.frame_tree_view_cmd_mes, text="+", width=10, command=self.add_measurement).grid(row=0, column=0)


        ttk.Button(self.frame_tree_view_cmd_mes, text="-", width=10, command=self.del_measurement).grid(row=0, column=1)


        ttk.Button(self.frame_tree_view_cmd_mes, text="Duplicate", width=10, command=self.duplicate_measurement).grid(row=0, column=2)



        self.frame_tree_view_cmd_burst = tk.LabelFrame(self.frame_cmd, text="Burst Analysis",
                                                     borderwidth=self.appearence_param.frameLabelBorderWidth)
        self.frame_tree_view_cmd_burst.grid(row=1, column=2)

        ttk.Button(self.frame_tree_view_cmd_burst, text="+", width=10, command=self.add_burst).grid(row=0, column=0)
        ttk.Button(self.frame_tree_view_cmd_burst, text="-", width=10, command=self.del_burst).grid(row=0, column=1)
        ttk.Button(self.frame_tree_view_cmd_burst, text="Duplicate", width=10, command=self.duplicate_burst).grid(row=0, column=2)



    def add_exp_file(self):
        self.controller.view.menu.askOpenSPC_file()

        # self.controller.update_all()

    def del_exp_file(self):
        pass

    def duplicate_exp_fil(self):
        pass

    def add_burst(self):
        #TODO name and comment dialog
        self.controller.launch_burst_analysis_GUI()

    def del_burst(self):
        pass

    def duplicate_burst(self):
        pass

    def insert_burst(self, burst):
        pass

    def insert_list_of_exp(self, dict_of_exp):
        """
        Mianly used for load state
        :param dict_of_exp:
        :return:
        """
        for exp_name, exp in dict_of_exp.items():
            iid_exp = self.insert_exp(exp)
            for mes_name, measurement in exp.measurements.items():
                self.insert_measurement(measurement, iid_exp)


    def insert_exp(self, exp):
        # iid = self.tree_view.insert(parent="", index='end', text=exp.file_name)
        nb_photon = exp.data.channels[0].nb_of_tick
        nb_of_channel = len(exp.data.channels)
        CPS = int(exp.data.channels[0].CPS)
        t_start_micros = exp.convert_ticks_in_seconds(exp.data.channels[0].start_tick) * 1E6
        t_end_micros =  exp.convert_ticks_in_seconds(exp.data.channels[0].end_tick) * 1E6
        iid = self.tree_view.insert(parent="", index='end', values=(exp.file_name, "", "",  nb_photon, CPS, nb_of_channel,  "", t_start_micros, t_end_micros))
        if exp.file_name not in self.exp_iid_dict:
            self.exp_iid_dict[exp.file_name] = iid
        self.tree_view.focus(iid)
        self.treeview_measurement_select(None)
        return iid

    def insert_measurement(self, measurement, parent_exp_iid=None):
        """
        Miunaly used for load state
        :param measurement:
        :param parent_exp_iid:
        :return:
        """
        if parent_exp_iid is None:
            parent_exp_iid = self.exp_iid_dict[self.controller.current_exp.file_name]

        if measurement.type == "burst":
            new_exp_name = self.controller.current_exp.file_name + "_burst"
            iid_burst = self.tree_view.insert(parent="", index='end',
                                        values=("", "", new_exp_name, measurement.type, measurement.nb_of_photon, "NA", measurement.num_channel,
                measurement.comment, measurement.start_tick, measurement.end_tick))
            self.tree_view.item(iid_burst, open=True)
            if measurement.name not in self.mes_iid_dict:
                self.mes_iid_dict[measurement.name] = iid_burst
            num = 0
            for burst in measurement.bursts:
                burst_name = "b_" + str(num)
                iid = self.tree_view.insert(parent=iid_burst, index='end',
                                            values=(
                                            "", burst_name, "", "", burst.nb_photon, burst.CPS,
                                            measurement.num_channel,
                                            "", burst.tick_start, burst.tick_end))
                self.tree_view.item(parent_exp_iid, open=True)
                if measurement.name not in self.mes_iid_dict:
                    self.mes_iid_dict[measurement.name + "_" + burst_name] = iid
                num += 1
        else:
            iid = self.tree_view.insert(parent=parent_exp_iid, index='end',
                                        values=("", "", measurement.name, measurement.type, measurement.nb_of_photon, "NA", measurement.num_channel,
                measurement.comment, measurement.start_tick, measurement.end_tick))

            self.tree_view.item(parent_exp_iid, open=True)
            if measurement.name not in self.mes_iid_dict:
                self.mes_iid_dict[measurement.name] = iid
        # self.tree_view.focus(iid)

        # self.treeview_measurement_select(None)


    def update_tree_view_line(self, measurement):
        iid = self.mes_iid_dict[measurement.name]
        self.tree_view.item(iid, values=("", "", measurement.name, measurement.type, measurement.nb_of_photon, "NA", measurement.num_channel,  measurement.comment, measurement.start_tick, measurement.end_tick))

    def update_tree_view(self):
        for mes_name, iid in self.mes_iid_dict.items():
            measurement = self.controller.get_measurement(mes_name)
            if measurement is not None:
                self.tree_view.item(iid, values=("",
                "", measurement.name, measurement.type, measurement.nb_of_photon, "NA", measurement.num_channel,
                measurement.comment, measurement.start_tick, measurement.end_tick))
        for exp_name, iid in self.exp_iid_dict.items():
            exp = self.controller.get_experiment(exp_name)
            nb_photon = exp.data.channels[0].nb_of_tick
            nb_of_channel = len(exp.data.channels)
            CPS = int(exp.data.channels[0].CPS)
            t_start_micros = exp.convert_ticks_in_seconds(exp.data.channels[0].start_tick) * 1E6
            t_end_micros = exp.convert_ticks_in_seconds(exp.data.channels[0].end_tick) * 1E6
            self.tree_view.item(iid, values=(exp.file_name, "", "", "",  nb_photon, CPS, nb_of_channel,  "", t_start_micros, t_end_micros))


    def add_measurement(self):
        if self.controller.current_exp is not None:
            d = add_measurement_dialog(self.master_frame, "add experiment", self.controller)
            if d.result is not None:
                mes_type, mes_name, mes_comment = d.result
                new_measurement = self.controller.create_measurement(mes_type, mes_name, mes_comment)
                self.controller.add_measurement(new_measurement)

                iid_current_exp = self.exp_iid_dict[self.controller.current_exp.file_name]

                iid = self.tree_view.insert(parent=iid_current_exp, index='end', values=("", "", mes_name, mes_type, "0", "NA", "NA", mes_comment, "NA", "NA"))
                self.tree_view.item(iid_current_exp, open=True)
                self.tree_view.focus(iid)
                self.mes_iid_dict[mes_name] = iid
                self.treeview_measurement_select(None)
        else:
            self.controller.log_message("No experiment loaded")

    def del_measurement(self):
        id_selected_item = self.tree_view.focus()
        selected_item = self.tree_view.item(id_selected_item)

        item_name_exp = selected_item["values"][0]
        item_name_mes = selected_item["values"][1]

        if item_name_exp in self.controller.model.experiments:
            # this is an experiment
            # exp = self.controller.set_current_measurement(item_name_exp)
            # self.controller.update_navigation()
            #TODO how to remove an exp ?
            pass
        elif item_name_mes in self.controller.current_exp.measurements:
            # this is a measurement
            self.controller.del_measurement(item_name_mes)
            self.tree_view.delete(id_selected_item)
            #TODO display nothing on the GUI


        # measurement_name = self.list_measurement.get(num)

    def clear_treeview(self):
        self.tree_view.delete(*self.tree_view.get_children())


    def duplicate_measurement(self):
        pass
        # list_selection = self.list_measurement.curselection()
        # num = int(list_selection[0])
        # measurement_name = self.list_measurement.get(num)
        # self.controller.duplicate_measurement(measurement_name)
        # self.list_measurement.insert(tk.END, measurement_name + "_b")

    def treeview_measurement_select(self, event):
        id_selected_item = self.tree_view.focus()
        selected_item = self.tree_view.item(id_selected_item)
        item_name_exp = selected_item["values"][0]
        item_name_mes = selected_item["values"][2]
        item_name_burst = selected_item["values"][1]

        if item_name_exp in self.controller.model.experiments:
            # this is an experiment
            exp = self.controller.set_current_exp(item_name_exp)
            if item_name_mes == "":
                # the user has selected a root exp without measurement
                # self.controller.set_current_measurement(None)
                self.controller.view.archi.analyze_area.display_measurement(None)
                # self.controller.update_all(is_full_update=True)
                return
        elif item_name_mes in self.controller.current_exp.measurements:
            # this is a measurement
            measurement = self.controller.set_current_measurement(item_name_mes)
            self.controller.view.archi.analyze_area.display_measurement(measurement)
            self.controller.display_measurement(measurement.name)
        elif item_name_burst != "":
            parent_iid = self.tree_view.parent(id_selected_item)
            if item_name_mes in self.controller.current_exp.measurements:
                measurement = self.controller.set_current_measurement(item_name_mes)
                num_burst = int(item_name_burst[2:])
                self.controller.display_burst(measurement.bursts[num_burst], measurement)

        elif item_name_mes in self.controller.current_exp.measurements:
            # this is a measurement
            measurement = self.controller.set_current_measurement(item_name_mes)
            self.controller.view.archi.analyze_area.display_measurement(measurement)
            self.controller.display_measurement(measurement.name)


#https://stackoverflow.com/questions/673174/file-dialogs-of-tkinter-in-python-3
from tkinter import filedialog, messagebox, simpledialog


class add_measurement_dialog(simpledialog.Dialog):
    """
    http://effbot.org/tkinterbook/tkinter-dialog-windows.htm
    """
    def __init__(self, master, title, controller):
        self.controller = controller
        super().__init__(master, title)

    def body(self, master):

        ttk.Label(master, text="Type:").grid(row=0)
        ttk.Label(master, text="Name :").grid(row=1)
        ttk.Label(master, text="Comment :").grid(row=2)

        self.cb_sv = tk.StringVar()
        self.cb = ttk.Combobox(master, width=25, justify=tk.CENTER, textvariable=self.cb_sv, values='')
        self.cb['values'] = ('FCS', 'lifetime', 'DLS', 'phosphorescence')
        self.cb.set('FCS')
        self.cb.bind('<<ComboboxSelected>>', self.change_type)
        self.cb.grid(row=0, column=1)

        self.e1_svar = tk.StringVar()
        self.e1 = ttk.Entry(master, textvariable=self.e1_svar)
        self.e2 = ttk.Entry(master)

        #default value
        default_type = "FCS"
        self.e1.insert(tk.END, default_type)
        self.e2.insert(tk.END, self.controller.get_available_name_for_measurement(default_type))

        self.e1.grid(row=1, column=1)
        self.e2.grid(row=2, column=1)

        self.result = None
        return self.e2 # initial focus

    def apply(self):
        first = self.cb_sv.get()
        second = self.e1.get()
        third = self.e2.get()
        self.result = first, second, third

    def change_type(self, event=None):
        self.e1_svar.set(self.controller.get_available_name_for_measurement(self.cb_sv.get()))


