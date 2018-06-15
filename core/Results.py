# from core.analyze import lifetime

class Results(object):

    def __init__(self):
        """
        Contains the current results from the analysis of the data
        """
        self.maxNbOfChannel = 8

        self.lifetimes = [None] * self.maxNbOfChannel  # lifetime.lifeTimeMeasurements()
        self.FCS_Measurements = [None] * self.maxNbOfChannel  # lifetime.lifeTimeMeasurements()
        self.DLS_Measurements = [None] * self.maxNbOfChannel  # lifetime.lifeTimeMeasurements()
        self.chronograms = [None] * self.maxNbOfChannel

        # self.timeZoomChronogram = None
        self.navigationChronogram = None

        self.mainPCH = None

        self.intensityHistogram = [None] * self.maxNbOfChannel
        self.correlationCurve = [None] * self.maxNbOfChannel

    def change_nb_of_detector(self):
        pass

    def add_channel(self):
        self.intensityHistogram.append(0)
        self.correlationCurve.append(0)
        self.chronograms.append(0)
        # self.microtimeHistograms.append(0)

class Chronogram():
    def __init__(self):
        self.xAxis = None
        self.data = None
        self.tickStart = None
        self.tickEnd = None
        self.nbOfBin = None
