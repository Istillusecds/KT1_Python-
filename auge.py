import numpy as np
import matplotlib.pyplot as plt


def auge(x: np.ndarray, w: int) -> tuple[plt.Figure, plt.Axes]:
    """
    Generates an eye diagram from the input signal.

    Parameters:
    -----------
    x : np.ndarray
        Vector of the sampled data signal.
    w : int
        Number of samples per symbol (T/TA). Must be an even integer.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the eye diagram.
    ax : matplotlib.axes.Axes
        The Axes object of the plotted eye diagram.


    Notes:
    ------
    Adapted from 'auge.m' (Kammeyer, 2000) for a NumPy-based implementation.
    """
    len: int = x.shape[-1]
    K: int = int(np.floor(len/w)-2)

    A: np.ndarray = np.zeros((2*w+1, K))
    for i in range(K):
        A[:, i] = x[i*w:(i+2)*w+1]

    t: np.ndarray = np.arange(start=-1, stop=1+1/w, step=1/w)

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, A, linestyle='-', linewidth=.5)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-3, 3])

    return fig, ax
