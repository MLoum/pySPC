import numpy as np
import os
from . import Results
# from .importFormat import bh_SPC_SET
# from .importFormat import pq_PT3_PT2
from .importFormat import pqreader
from .importFormat import bhreader
from .importFormat import nist_fpga
from .analyze import bin


from .analyze import lifetime, FCS, DLS


class Channel():
    def __init__(self, name=""):
        self.name = name
        self.photons = []
        self.nbOfTick = 0
        self.startTick = 0
        self.endTick = 0
        self.CPS = 0

class Data():
    """
    Contains the SPC raw data and analysis tools
    """

    def __init__(self, results, expParam):

        #Ici cela serait du Data-Oriented Design
        # http://gamesfromwithin.com/data-oriented-design
        #https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html
        #self.photonDataType = np.dtype([('timestamps', np.uint64), ('nanotimes', np.uint32) , ('detectors', np.uint16), ('isFiltered', np.bool_)])
        self.photonDataType = np.dtype([('timestamps', np.uint64), ('nanotimes', np.uint32), ('detectors', np.uint16)])

        #Je ne fait pas une approche 100% objet avec une classe detecteur et une classe photon, car cela complique bcp l'utilisation de numpy
        self.timestamps = []
        self.nanotimes = []
        self.isFiltered = []

        self.channels = []

        self.results = results
        self.expParam = expParam

    def delData(self):
        del self.timestamps[:]
        del self.nanotimes[:]
        del self.isFiltered[:]

    def loadFromFile(self, filePath):
        filename, file_extension = os.path.splitext(filePath)
        if file_extension == ".spc" :
            timestamps, detectors, nanotimes, timestamps_unit, meta = self.loadSPC_Bh_File(filePath)
            #Done inside the loadSPC_Bh_File function
            self.expParam.fillWithSPCMetaData(meta, timestamps_unit)

        elif file_extension == ".pt3" :
            timestamps, detectors, nanotimes, meta = pqreader.load_pt3(filePath)
            self.expParam.fillWithPt3MetaData(meta)

        elif file_extension == ".ttt" :
            timestamps, detectors, nanotimes, timestamps_unit, meta = nist_fpga.load_ttt(filePath)
            self.expParam.fillWith_ttt_MetaData(meta)


        # self.photons = np.empty(len(timestamps), self.photonDataType)
        # self.photons['timestamps'] = timestamps
        # self.photons['nanotimes'] = nanotimes
        # self.photons['detectors'] = detectors
        #On va plutot utiliser les masques
        #self.photons['isFiltered'] = np.zeros(len(timestamps), np.bool_)

        #Les photons ne sont pas triès par detecteur, il le sont par ordre d'arrivée
        unique, return_index, unique_inverse, unique_counts  = np.unique(detectors, return_index = True, return_inverse  = True, return_counts = True)

        #print(unique, unique_counts)

        soft_channel_value = 0

        numChannel = 0
        nbOfChannel = 0
        for value in unique:
            if unique_counts[numChannel] > 50 and value >= 0:
                nbOfChannel += 1
            numChannel +=1

        self.expParam.nbOfChannel = nbOfChannel

        self.delData()
        del self.channels[:]

        numChannel = 0
        soft_channel_value = 0
        for value in unique:

            #50 is arbiratry, we do this to filter false count on some detector.
            if unique_counts[numChannel] < 50:
                numChannel += 1
                soft_channel_value += 1
                continue

            # On créé un masque et on fait une copie des elements non masqués vers le channel.
            m_ = np.ma.masked_where(detectors != value, timestamps)
            timestampsMasked = np.ma.compressed(m_)

            m_ = np.ma.masked_where(detectors != value, nanotimes)
            nanotimesMasked = np.ma.compressed(m_)

            photons = np.empty(unique_counts[soft_channel_value], self.photonDataType)

            photons['timestamps'] = timestampsMasked
            photons['nanotimes'] = nanotimesMasked

            # self.isFiltered.append(np.ones(unique_counts[soft_channel_value], dtype=bool))
            # soft_channel_value += 1


            #Mask an array where a condition is met.
            # condition : array_like
            #print(np.shape(self.photons))
            # print(self.photons['detectors'])

            # m_ = np.ma.masked_where(self.photons['detectors']  != i, self.photons)
            # print(m_)
            # print(m_['detectors'])
            #
            # print(np.ma.getmask(m_))
            # test = np.ma.compressed(m_)
            # print(test)

            #Count the non-masked elements of the array along the given axis.
            #nbElementNonMasked = np.ma.MaskedArray.count(m_['detectors'])

            c = Channel()
            c.photons = photons
            c.startTick = c.photons['timestamps'][0]
            c.endTick = c.photons['timestamps'][-1]
            c.nbOfTick = c.photons['timestamps'].size
            c.CPS = float(c.nbOfTick) / (c.endTick - c.startTick) / self.expParam.mAcrotime_clickEquivalentIn_second
            self.channels.append(c)
            self.results.addChannel()
            # TODO  ???
            soft_channel_value += 1
            numChannel += 1


            #grande question : quand est-il des tableaux avec des masques, ici on garderait une version masquée
            # je prefere garder une copie.
            #c.photons.append[np.ma.masked_where(self.photons['detectors']  == i, self.photons)]

        #J'utilise d'abord np.unique sur le detector qui m'indique quelle valeur de detecteur sont présent dans le fichier et où.
        # Je peux par exemple savoir que trois detecteurs ont été utilisés : 0,1 et 3 et je connais l'index de leur tick respectif (unique_inverse).

        # for value in unique:
        #     m_ = np.ma.masked_where(unique_inverse == return_index[soft_channel_value], timestamps)
        #     self.timestamps.append(np.ma.compressed(m_))
        #
        #     m_ = np.ma.masked_where(unique_inverse == return_index[soft_channel_value], nanotimes)
        #     self.nanotimes.append(np.ma.compressed(m_))
        #
        #     self.isFiltered.append(np.ones(unique_counts[soft_channel_value], dtype=bool))
        #     soft_channel_value += 1


    def newGeneratedExp(self, type, params):
        if type == "Poisson":
            self.delData()
            del self.channels[:]

            self.expParam.nbOfMicrotimeChannel = 256
            #TODO ask user ?
            self.expParam.mAcrotime_clickEquivalentIn_second = 50E-9
            self.expParam.mIcrotime_clickEquivalentIn_second = 10E-12

            time_s = params[0]
            time_tick = time_s / self.expParam.mAcrotime_clickEquivalentIn_second

            count_per_second_s = params[1]
            count_per_tick = count_per_second_s * self.expParam.mAcrotime_clickEquivalentIn_second

            timeStamps = self.generatepoissonNoise(count_per_tick, 0, time_tick)
            photons = np.empty(np.size(timeStamps), self.photonDataType)

            photons['timestamps'] = timeStamps
            photons['nanotimes'] = 0

            c = Channel()
            c.photons = photons
            c.startTick = c.photons['timestamps'][0]
            c.endTick = c.photons['timestamps'][-1]
            c.nbOfTick = c.photons['timestamps'].size
            c.CPS = float(c.nbOfTick) / (c.endTick - c.startTick) / self.expParam.mAcrotime_clickEquivalentIn_second
            self.channels.append(c)



    def loadSPC_Bh_File(self, filePath):
        #find associate set
        path_set = os.path.splitext(filePath)[0] + '.set'
        if(os.path.isfile(path_set)):
                meta = bhreader.load_set(path_set)
                self.expParam.fillWithSPCMetaData(meta, None)
        else :
            return "default set File missing"

        #TODO fill ExpParam with set
        timestamps, detector, nanotime, timestamps_unit = bhreader.load_spc(filePath, spc_model="SPC-130")
        return timestamps, detector, nanotime, timestamps_unit, meta

    def searchIdxOfPhotonWithTimeSupTo_t1_and_InfTo_t2(self, array, t1, t2):
        #Find indices where elements should be inserted to maintain order
        return np.searchsorted(array, (t1,t2))
        pass

    #TODO put into a file in analyze and call it Bin.
    def chronogram(self, numChannel, startTick, endTick, binInTick):
        """
        The x axis is in ->microsecond<-
        """
        timeStamps = self.channels[numChannel].photons['timestamps']

        startTick = np.uint64(startTick)
        endTick = np.uint64(endTick)

        #TODO Expliquer le +1, je pense que c'est du à des pb de valeurs arrondies... Au pire la derniere case est vide.
        nbOfBin = int((endTick - startTick)/ binInTick) + 1
        # Find indices where elements should be inserted to maintain order
        idxStart, idxEnd = np.searchsorted(timeStamps, (startTick, endTick))

        timesStamps = np.copy(timeStamps[idxStart:idxEnd])
        timesStamps -= startTick

        numStartBin = int (startTick / binInTick)
        #numEndBin = int( endTick / binInTick)
        numEndBin = numStartBin + nbOfBin

        #FIXME moins de divisions, ici on prend tout le fichier
        # numOfBinForEachPhoton =  timesStamps / binInTick
        # numOfBinForEachPhoton = numOfBinForEachPhoton.astype(int)

        # #Default value for "range" seems fine  -> The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()). Values outside the range are ignored.
        # self.results.mainChronogram = Results.Chronogram()
        # self.results.mainChronogram.tickStart, self.results.mainChronogram.tickEnd, self.results.mainChronogram.nbOfBin = startTick, endTick, nbOfBin
        # self.results.mainChronogram.data, self.results.mainChronogram.xAxis  =  np.histogram(numOfBinForEachPhoton, nbOfBin)
        # #time axis in millisecond
        # self.results.mainChronogram.xAxis *= binInTick * self.expParam.mAcrotime_clickEquivalentIn_second*1E6 #ms
        # #We have to cut by one element the x axis because it is on element longer than the data (NB : there is no copy, just "a view")
        # self.results.mainChronogram.xAxis = self.results.mainChronogram.xAxis[:-1]


        chronogram = Results.Chronogram()
        chronogram.tickStart, chronogramtickEnd, chronogram.nbOfBin = startTick, endTick, int(nbOfBin)
        #chronogram.data, chronogram.xAxis = np.histogram(numOfBinForEachPhoton, int(nbOfBin))

        chronogram.data = np.zeros(chronogram.nbOfBin + 1, dtype=np.int)
        bin.bin(timesStamps, chronogram.data, binInTick)
        #chronogram.data
        #TODO UNDERSTAND We have to cut by one element the x axis because it is on element longer than the data (NB : there is no copy, just "a view")
        chronogram.data = chronogram.data[:-1]

        chronogram.xAxis = np.arange(numStartBin, numEndBin, dtype=np.float64)
        chronogram.xAxis *= binInTick
        chronogram.xAxis += chronogram.tickStart
        chronogram.xAxis *= self.expParam.mAcrotime_clickEquivalentIn_second * 1E6


        #time axis in millisecond
        # chronogram.xAxis += chronogram.tickStart
        # chronogram.xAxis *= binInTick * self.expParam.mAcrotime_clickEquivalentIn_second * 1E6  # microsecond
        # We have to cut by one element the x axis because it is on element longer than the data (NB : there is no copy, just "a view")
        # chronogram.xAxis = chronogram.xAxis[:-1]

        return chronogram

    def PCH(self, chronogram):
        self.results.mainPCH = Results.PCH()
        self.results.mainPCH.data, self.results.mainPCH.xAxis = np.histogram(chronogram.data, chronogram.data.max())

        self.results.mainPCH.nbOfBin = len(self.results.mainPCH.data)

        # We have to cut by one element the x axis because it is on element longer than the data (NB : there is no copy, just "a view")
        self.results.mainPCH.xAxis = self.results.mainPCH.xAxis[:-1]


    def timeDifference(self, numChannel):
        self.results.timeDifference[numChannel] = np.diff(self.timestamps[numChannel])

    def microTimeLifeTime(self, numChannel, startTick, endTick):
        timeStamps = self.channels[numChannel].photons['timestamps']
        nanotimes = self.channels[numChannel].photons['nanotimes']
        idxStart, idxEnd = np.searchsorted(timeStamps, (startTick, endTick))
        nanotimes = nanotimes[idxStart:idxEnd]

        if self.results.lifeTimeMeasurements[numChannel] == None:
            self.results.lifeTimeMeasurements[numChannel] = lifetime.lifeTimeMeasurements()

        self.results.lifeTimeMeasurements[numChannel].createHistogramm(nanotimes, self.expParam.nbOfMicrotimeChannel, self.expParam.mIcrotime_clickEquivalentIn_second)


    def FCS(self, numChannel_1, numChannel_2, startTick, endTick, maxCorrelationTime_ms=1000):
        #TODO cross correlation
        numChannel = numChannel_1

        timeStamps = self.channels[numChannel].photons['timestamps']
        nanotimes = self.channels[numChannel].photons['nanotimes']
        idxStart, idxEnd = np.searchsorted(timeStamps, (startTick, endTick))
        timeStamps_reduc = timeStamps[idxStart:idxEnd]

        if self.results.FCS_Measurements[numChannel] == None:
            self.results.FCS_Measurements[numChannel] = FCS.FCSMeasurements()

        maxCorrelationTimeInTick = int (maxCorrelationTime_ms/1000.0 / self.expParam.mAcrotime_clickEquivalentIn_second)
        self.results.FCS_Measurements[numChannel].correlateMonoProc(timeStamps_reduc, timeStamps_reduc,  maxCorrelationTimeInTick)

    def DLS(self, numChannel_1, numChannel_2, startTick, endTick, maxCorrelationTime_ms=1000):
        numChannel = numChannel_1

        timeStamps = self.channels[numChannel].photons['timestamps']
        nanotimes = self.channels[numChannel].photons['nanotimes']
        idxStart, idxEnd = np.searchsorted(timeStamps, (startTick, endTick))
        timeStamps_reduc = timeStamps[idxStart:idxEnd]

        if self.results.DLS_Measurements[numChannel] == None:
            self.results.DLS_Measurements[numChannel] = DLS.DLS_Measurements()

        maxCorrelationTimeInTick = int (maxCorrelationTime_ms/1000.0 / self.expParam.mAcrotime_clickEquivalentIn_second)
        self.results.DLS_Measurements[numChannel].correlateMonoProc(timeStamps_reduc, timeStamps_reduc,  maxCorrelationTimeInTick)


    def generatepoissonNoise(self, meanRateInTick, t_start_click, t_end_click):
        """
        time in click
        :param meanRateInTick: lambda of poissonian law.
        :param t_start:
        :param t_end:
        :return:
        """
        #http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/
        #ratePerClick = ratePerS / self.expParam.mAcrotime_clickEquivalentIn_second
        #TODOname of ratePerClick
        nbOfTickToGenerate = int((t_end_click - t_start_click)*meanRateInTick)
        arrivalTimes = t_start_click +  np.cumsum(-(np.log(1.0 - np.random.random(nbOfTickToGenerate)) / meanRateInTick).astype(np.uint64))
        lastSample = np.searchsorted(arrivalTimes, t_end_click)
        return arrivalTimes[:lastSample]
        #nbCorrelationPoint = int(maxCorrelationTime_s / self.expParam.mAcrotime_clickEquivalentIn_second)


