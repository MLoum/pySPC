
class PlotParam():
    def __init__(self):
        self.graph_result = GraphResultPlotParam()


class GraphResultPlotParam():
    def __init__(self):
        self.is_plot_error_bar = True
        self.is_plot_all_FCS_curve = False
        self.is_autoscale = True
        self.is_zoom_x_selec = False
        self.is_plot_fit = True