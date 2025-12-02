import numpy as np
import numpy.fft as fft
from typing import Tuple


def nextpow2(n1: int) -> int:
    """computes the next exponent that for which
    n1 <= 2**n
    is satisfied
    :parameter
    n1: float
        number, whose supremum shall be a power of 2
    :return
    exp: int
        the required power factor"""

    return int(np.ceil(np.log2(n1)))


def fftseq(m: np.ndarray, ts: float, df_req: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """The sequence is zero padded to meet the required frequency resolution df.
    ts is the sampling interval. The output df is the final frequency resolution.
    Output m is the zero padded version of input m. M is the FFT.
    :parameter
    m: np.ndarray
        the sequence of which the FFT is computed
    ts: float
        sampling interval
    df_req: float
        required frequency resolution. Default is None

    :return
    M: np.ndarray
        FFT of sequence m
    m_zp: np.ndarray
        zero-padded version of the input sequence m
    df: float
        the actual frequency resolution of M
    """
    n1: float = 0
    fs: float = 1 / ts
    if not df_req is None:
        n1 += fs / df_req

    n2 = m.size
    n: int = int(2 ** max(nextpow2(n1), nextpow2(n2)))
    M: np.ndarray = fft.fft(m, n)
    m_zp: np.ndarray = np.hstack([m, np.zeros(n - n2)])
    df: float = fs / n

    return M, m_zp, df
