import numpy as np


def load_ttt(filename):

    mainClock_MHz = 48.0
    timestamps_unit = 1/mainClock_MHz * 1e-6


    """
    Bit:     26-0      27    28    29  30    31
    Meaning: Timestamp Ch. 1 Ch. 2 Ch. 3 Ch. 4 Counter cleared
    """

    f = open(filename, 'rb')

    np.fromfile(f, dtype='u4', count=1)[0]

    # ...and then the remaining records containing the photon data
    # ttt_dtype = np.dtype([('timeStamp', '<u2'), ('ch1', '<u2'), ('ch2', '<u2'), ('ch3', '<u2'), ('ch4', '<u2'), ('overflow', '<u2')])
    data = np.fromfile(f, dtype=np.int32)

    #fake microtime
    nanotime = np.zeros(np.size(data))

    """
    The Start bit is high when the user inputs a TTL pulse on the Start channel (pin 97), or when the internal clock resets, which 
    happens every ≈ 2.796 s when using a 48 MHz. The second piece of each line is an integer representing the
    number of clock cycles since the last Start event.
    """

    # Build the macrotime
    # 0x07FF FFFF is 0000 0111 1111  1111 1111 1111 1111
    timestamps = np.bitwise_and(data, 0x07FFFFFF).astype(dtype='int64')

    d1 = np.bitwise_and(data, 0x08000000)/0x08000000
    d2 = np.bitwise_and(data, 0x10000000)/0x10000000 * 2
    d3 = np.bitwise_and(data, 0x20000000)/0x20000000 * 3
    d4 = np.bitwise_and(data, 0x40000000)/0x40000000 * 4

    nb_d2 = np.count_nonzero(d2)
    nb_d3 = np.count_nonzero(d3)

    detector = d1 + d2 + d3 + d4
    detector -= 1

    #x8000 is 1000 0000 0000 0000   0000 0000 0000 0000 in binary
    overflow = (np.bitwise_and(data, 0x80000000)/0x80000000).astype(dtype='int64')

    nb_overflow = np.count_nonzero(overflow)

    # Each overflow occurs every 2^27 macrotimes
    #overflow = np.cumsum(overflow)
    #overflow = np.left_shift(np.cumsum(overflow), 27)
    overflow = np.left_shift(np.cumsum(overflow), 27)

    # Add the overflow bits
    timestamps += overflow

    timestamps -= timestamps[0]

    meta = ""

    return timestamps, detector, nanotime, timestamps_unit, meta