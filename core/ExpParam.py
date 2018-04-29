import numpy as np


class ExperimentParam(object):
    def __init__(self):
        self.meta = None
        self.nbOfChannel = None
        self.nbOfMicrotimeChannel = None
        self.mAcrotime_clickEquivalentIn_second = None
        self.mIcrotime_clickEquivalentIn_second = None
        self.spcCountingCardType = None
        self.spcFileType = None

    def fillWithPt3MetaData(self, meta):
        self.meta = meta
        self.mAcrotime_clickEquivalentIn_second = meta['timestamps_unit'][0]
        self.mIcrotime_clickEquivalentIn_second = meta['nanotimes_unit'][0]

        self.nbOfMicrotimeChannel = meta['header'][0][-8]
        #FIXME
        #self.nbOfChannel = 65535
        self.spcFileType  = 'pt3'

    def fillWithSPCMetaData(self, meta, timeStampUnits):
        self.meta = meta

        #TODO trouver les bonnes clefs dans le meta.
        self.mIcrotime_clickEquivalentIn_second = meta["sys_params"]["SP_TAC_TC"]
        self.mAcrotime_clickEquivalentIn_second = timeStampUnits


        self.nbOfMicrotimeChannel = meta["sys_params"]["SP_ADC_RE"]
        self.nbOfChannel = 1

        self.spcFileType  = 'spc'

    def fillWith_ttt_MetaData(self, meta):
        self.meta = meta
        self.mAcrotime_clickEquivalentIn_second = 1/48.0 * 1E-6
        self.mIcrotime_clickEquivalentIn_second  = 1e-9 #arbitrary