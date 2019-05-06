import numpy as np


def load_ptn(filename):

    mainClock_MHz = 20
    timestamps_unit = 1/mainClock_MHz * 1e-6


    """
    Bit:     40-0      40-56     56-64
    Meaning: Timestamp Microtime Channel
    """

    f = open(filename, 'rb')

    # np.fromfile(f, dtype='u4', count=1)[0]

    # ...and then the remaining records containing the photon data
    # ttt_dtype = np.dtype([('timeStamp', '<u2'), ('ch1', '<u2'), ('ch2', '<u2'), ('ch3', '<u2'), ('ch4', '<u2'), ('overflow', '<u2')])
    data = np.fromfile(f, dtype=np.uint64)

    # 0x000FFFFF is 0000 0000 1111  1111 1111 1111 1111
    timestamps = np.bitwise_and(data, 0x000FFFFF).astype(dtype='uint64')
    # 0x0FF00000
    nanotime = np.bitwise_and(data, 0x0FF00000).astype(dtype='uint16')
    detector = np.bitwise_and(data, 0xF0000000).astype(dtype='uint8')


    meta = ""

    return timestamps, detector, nanotime, timestamps_unit, meta