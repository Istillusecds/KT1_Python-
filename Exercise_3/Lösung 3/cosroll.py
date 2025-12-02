import numpy as np


def cosroll(r: float, w: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a cosine roll-off shaped signal.

    Parameters:
    -----------
    r : float
        Roll-off factor.
        For r = 0, a truncated sinc function (cut at ±T·L/2) is used.

    w : int
        Number of samples per symbol (T/TA). Must be an even integer.

    L : int
        Number of symbol periods (T).
        Condition: L must be even and L ≥ 1 / r.

    Returns:
    --------
    t : np.ndarray
        Time vector, normalized to the symbol period T.
        Range: t/T = -L/2 to +L/2.

    g : np.ndarray
        Cosine roll-off pulse shape.

    Notes:
    ------
    Adapted from 'cosroll.m' (Kammeyer, 1990) for a NumPy-based implementation.
    """

    LL: int = w * L
    t: np.ndarray = np.arange(-LL/2, LL/2+1)/w

    if r == 0:
        g: np.ndarray = np.sinc(t)
    else:
        np.seterr(divide='ignore')
        g: np.ndarray = np.sinc(t) * np.cos(r * np.pi * t) / (1 - (2 * r * t)**2)
        np.seterr(divide='warn')
        g[np.logical_or(np.abs(g) == np.inf, np.isnan(g))] = r / 2 * np.sin(np.pi / (2 * r))

    return t, g
