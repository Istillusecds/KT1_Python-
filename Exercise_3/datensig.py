import numpy as np


def datensig(g: np.ndarray, w: int, d: np.ndarray, A: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a shaped data signal using pulse shaping.

    Parameters:
    -----------
    g : np.ndarray
        Transmit pulse (sampling frequency fA = W/T),
        e.g., generated via `cosroll` or `wurzcos`.

    w : int
        Number of samples per symbol interval (T/TA).

    d : np.ndarray
        Input data vector (complex values allowed).

    A : bool, optional
        If True: pre- and post-ringing (transient) sections are removed.
        The first and last L/2 values of the input data vector `d`
        are treated as irrelevant (padding).
        Example: for L = 4, the input would look like:
        `d = [d(*) d(*) d(1) d(2) ... d(N) d(*) d(*)]`
        and the corresponding reference vector will be:
        `d_ref = [d(1) d(2) ... d(N)]`.

    Returns:
    --------
    d_ref : np.ndarray
        The reference data vector for bit error rate (BER) comparison.
        When `A=True`, this is the input vector `d` with pre- and post-padding trimmed.

    x : np.ndarray
        The oversampled shaped data signal.

    Notes:
    ------
    The comparison between `d_ref` and `np.sign(x[::w])`
    directly allows for Bit Error Rate (BER) evaluation.
    """
    dd: np.ndarray = np.zeros(w * d.shape[-1])
    dd[0::w] = d

    x: np.ndarray = np.convolve(dd, g, mode='full')

    if A:
        L: float = (g.shape[-1]-1)/w
        K: int = np.floor(L/2).__int__()
        x = x[K * w: -K * w]        # TODO check, deviates from MATLAB but works fine?

    return d, x
