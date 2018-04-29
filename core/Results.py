from .analyze import lifetime



class Results(object):

    def __init__(self):
        """
        Contains the current results from the analysis of the data
        """
        self.maxNbOfChannel = 8

        self.lifeTimeMeasurements = [None] * self.maxNbOfChannel  #lifetime.lifeTimeMeasurements()
        self.FCS_Measurements = [None] * self.maxNbOfChannel  # lifetime.lifeTimeMeasurements()
        self.DLS_Measurements = [None] * self.maxNbOfChannel  # lifetime.lifeTimeMeasurements()
        self.chronograms = [None] * self.maxNbOfChannel

        # self.timeZoomChronogram = None
        self.navigationChronogram = None

        self.mainPCH = None

        self.intensityHistogram = [None] * self.maxNbOfChannel
        self.correlationCurve = [None] * self.maxNbOfChannel


    def changeNbOfDetector(self):
        pass

    def addChannel(self):
        self.intensityHistogram.append(0)
        self.correlationCurve.append(0)
        self.chronograms.append(0)
        #self.microtimeHistograms.append(0)



class Chronogram():
    def __init__(self):
        xAxis = None
        data = None
        tickStart = None
        tickEnd = None
        nbOfBin = None

class FCS_Curve():
    def __init__(self):
        xAxis = None
        data = None
        fitCurve = None
        stdDev = None

class PCH():
    def __init__(self):
        xAxis = None
        data = None
        fitCurve = None
        stdDev = None
        nbOfBin = None