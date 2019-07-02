
class appearenceParam():
    def __init__(self):
        self.frameLabelBorderWidth = 7
        self.IR_trace = "k--"


        self.line_type_fit_lifetime = "b-"
        self.line_type_fit_FCS = "b-"
        self.line_type_fit = "b-"

        self.line_type_data = "ro"
        self.line_type_data_non_selected = "bo"
        self.alpha_data = 0.5
        self.alpha_selected = 0.7
        self.alpha_non_selected = 0.4
        self.line_type_residual = "k"

        # Result area
        self.font_xy_coordinate = 'Helvetica 18 bold'

        # Graph result
        self.graph_result_shift_amount = 0.07
        self.alpha_error_bar = 0.2


    def saveToIniFile(self):
        pass