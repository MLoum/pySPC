import tkinter as tk
from tkinter import ttk
import queue

import tkinter.scrolledtext as tkst

import logging
from GUI.graph.Graph_Results import Graph_Results

class QueueHandler(logging.Handler):
    """Class to send logging records to a queue

    It can be used from different threads
    """

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)



class TextHandler(logging.Handler):
    # This class allows you to log to a Tkinter Text or ScrolledText widget
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            # Autoscroll to the bottom
            self.text.yview(tk.END)
        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)


class Log_area():
    def __init__(self, masterFrame, view, controller, appearenceParam):
        self.master_frame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = appearenceParam
        self.time_polling = 300


    def populate(self):

        # Add text widget to display logging info
        self.log_area = tkst.ScrolledText(self.master_frame, state='disabled')
        self.log_area.configure(font='TkFixedFont')
        self.log_area.grid(column=0, row=0, sticky='w', columnspan=4)

        # Create textLogger
        self.text_handler = TextHandler(self.log_area)

        # Logging configuration
        logging.basicConfig(filename='test.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')

        # Add the handler to logger


        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
        self.text_handler.setFormatter(formatter)
        self.logger.addHandler(self.text_handler)



        # self.scrolled_text = tkst.ScrolledText(self.master_frame, state='disabled', height=12)
        # self.scrolled_text.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        # self.scrolled_text.configure(font='TkFixedFont')
        # self.scrolled_text.tag_config('INFO', foreground='black')
        # self.scrolled_text.tag_config('DEBUG', foreground='gray')
        # self.scrolled_text.tag_config('WARNING', foreground='orange')
        # self.scrolled_text.tag_config('ERROR', foreground='red')
        # self.scrolled_text.tag_config('CRITICAL', foreground='red', underline=1)
        # # Create a logging handler using a queue
        # self.log_queue = queue.Queue()
        # self.queue_handler = QueueHandler(self.log_queue)
        # formatter = logging.Formatter('%(asctime)s: %(message)s')
        # self.queue_handler.setFormatter(formatter)
        #
        # self.logger = logging.getLogger()
        # self.logger.setLevel(logging.INFO)
        #
        # self.logger.addHandler(self.queue_handler)
        #
        # # Start polling messages from the queue
        # self.master_frame.after(self.time_polling, self.poll_log_queue)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        self.view.logger = self.logger

        self.logger.info('Hello')



        self.analyze_pgb = ttk.Progressbar(self.master_frame, orient="horizontal", length=500, mode='indeterminate')
        self.analyze_pgb.grid(row=1, column=0)

    # def display(self, record):
    #     msg = self.queue_handler.format(record)
    #     self.scrolled_text.configure(state='normal')
    #     self.scrolled_text.insert(tk.END, msg + '\n', record.levelname)
    #     self.scrolled_text.configure(state='disabled')
    #     # Autoscroll to the bottom
    #     self.scrolled_text.yview(tk.END)
    #
    # def poll_log_queue(self):
    #     # Check every self.time_polling ms if there is a new message in the queue to display
    #     while True:
    #         try:
    #             record = self.log_queue.get(block=False)
    #         except queue.Empty:
    #             break
    #         else:
    #             self.display(record)
    #     self.master_frame.after(self.time_polling, self.poll_log_queue)




    def add_log_message(self, msg, level="info"):
        self.logger.info(msg)
