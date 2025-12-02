import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import os

from cosroll import cosroll
from datensig import datensig
from auge import auge
for g in ("kt1.mplstyle", "../kt1.mplstyle", "../../kt1.mplstyle"):
    try:
        plt.style.use(g)
        break
    except OSError:
        continue
else:
    raise FileNotFoundError("kt1.mplstyle nicht gefunden in ., .. oder ../..")


def exercise_one() -> None:
    """
    Solution of Tutorial 03, Exercise 01
    """
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(wspace=.5, hspace=.6)
    fig.set_size_inches(4, 3)

    w = 4   # Samples per Symbol Duration T

    # Calculate and plot impulse response for roll-off factor r=0.5
    t1, g1 = cosroll(r=0.5, w=w, L=8)
    ax[0, 0].stem(t1, g1, 'b-', markerfmt='o', basefmt="black")
    markerline, _, _ = ax[0, 0].stem(t1[0:16:4], g1[0:16:4], markerfmt='x', basefmt=" ")
    markerline.set_markersize(6)
    markerline, _, _ = ax[0, 0].stem(t1[20::4], g1[20::4], markerfmt='x', basefmt=" ")
    markerline.set_markersize(6)
    ax[0, 0].set_xlim([-4, 4])
    ax[0, 0].set_ylabel('$g_1(t)$')
    ax[0, 0].set_xlabel('Normierte Zeit $\\frac{t}{T}$')
    ax[0, 0].set_ylim([-.2, 1.2])
    ax[0, 0].set_title('Impulsantwort, r=0.5')

    # Calculate and plot transfer function for r = 0.5
    G1 = fft.fftshift(fft.fft(g1, n=64, norm='ortho')).__abs__()     # Increase resolution through zero-padding -> n=64
    f1 = fft.fftshift(fft.fftfreq(n=G1.shape[-1], d=1/w))
    ax[0, 1].plot(f1, G1)
    ax[0, 1].set_xlim([0, 1.5])
    ax[0, 1].set_ylabel('$|G_1(f)|$')
    ax[0, 1].set_xlabel('Normierte Frequenz $f \\cdot T$')
    ax[0, 1].set_ylim([0, 1])
    ax[0, 1].set_title('Übertragungsfunktion, r=0.5')

    # Annotate plot (not required in task)
    ax[0, 1].plot([.25, .25], [.7, .3], [.75, .75], [.7, 0.], linestyle='dashed', color='black', linewidth=0.5)
    ax[0, 1].annotate('', (.25, 0.5), (.75, 0.5),
                      arrowprops=dict(arrowstyle='<-', linewidth=0.5, shrinkA=0, shrinkB=0))
    ax[0, 1].annotate('Nyquistflanke', xy=[.78, .5])

    # Calculate and plot impulse response for roll-off factor r=1
    t2, g2 = cosroll(r=1, w=w, L=4)
    ax[1, 0].stem(t2, g2, 'b-', basefmt="black")
    markerline, _, _ = ax[1, 0].stem(t2[0:8:4], g2[0:8:4], markerfmt='x', basefmt=" ")
    markerline.set_markersize(6)
    markerline, _, _ = ax[1, 0].stem(t2[12::4], g2[12::4], markerfmt='x', basefmt=" ")
    markerline.set_markersize(6)
    markerline, _, _ = ax[1, 0].stem(t2[14], g2[14], markerfmt='yx', basefmt=" ")
    markerline.set_markersize(6)
    markerline, _, _ = ax[1, 0].stem(t2[2], g2[2], markerfmt='yx', basefmt=" ")
    markerline.set_markersize(6)
    ax[1, 0].set_xlim([-2, 2])
    ax[1, 0].set_ylabel('$g_2(t)$')
    ax[1, 0].set_xlabel('Normierte Zeit $\\frac{t}{T}$')
    ax[1, 0].set_ylim([-.2, 1.2])
    ax[1, 0].set_title('Impulsantwort, r=1')

    # Calculate and plot transfer function for r = 1
    G2 = fft.fftshift(fft.fft(g2, n=64, norm='ortho')).__abs__()
    f2 = fft.fftshift(fft.fftfreq(n=G2.shape[-1], d=1 / w))
    ax[1, 1].plot(f2, G2)
    ax[1, 1].set_xlim([-4, 4])
    ax[1, 1].set_ylabel('$|G_2(f)|$')
    ax[1, 1].set_xlabel('Normierte Frequenz $f \\cdot T$')
    ax[1, 1].set_xlim([0, 1.5])
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].set_title('Übertragungsfunktion, r=1')

    # Annotate plot (not required in task)
    x1, y1 = [1, 1], [0, 5]
    ax[1, 1].plot(x1, y1, linestyle='dashed', color='black', linewidth=0.5)
    ax[1, 1].annotate('', (0, 0.5), (1, 0.5),
                      arrowprops=dict(arrowstyle='<-', linewidth=0.5, shrinkA=0, shrinkB=0))
    ax[1, 1].annotate('Nyquistflanke', xy=[1, 1], xytext=[0.25, .55])

    fig.suptitle('Aufgabe 1: Entwurf von Kosinus-roll-off-Filtern')

    plt.show()


def exercise_two() -> None:
    """
    Solution of Tutorial 03, Exercise 02
    """

    """ Exercise 2.1 """
    # data sequence
    d: np.ndarray = np.array([1, -1, 1, 1, -1, -1])

    # cos-roll-off impulse response
    w: int = 8
    t: np.ndarray
    g: np.ndarray
    t, g = cosroll(r=0.5, w=w, L=8)

    # zero-stuffing
    dd: np.ndarray = np.zeros(w*d.shape[-1])
    dd[0::w] = d

    # convolution
    x: np.ndarray = np.convolve(dd, g)

    # plot output
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.25)
    tx: np.ndarray = np.arange(0, x.shape[-1])/w + t.min()
    plt.plot(tx, x)
    plt.xlim([round(tx.min()), round(tx.max())])
    plt.xlabel('Normierte Zeit t/T')
    plt.ylabel('Ausgangssignal x(t)')
    plt.suptitle('Aufgabe 2.1: Impulsformung')
    plt.ylim([-2, 2])

    # sample at time instants i*T
    ti: np.ndarray = tx[0::w]
    xi: np.ndarray = x[0::w]
    plt.stem(ti, xi, 'b-', markerfmt='o', basefmt=' ')
    plt.show()

    """ Exercise 2.2 """
    N: int = 10000
    d: np.ndarray = np.sign(np.random.randn(N))

    def calculate_ISI(d: np.ndarray, r: float, L: int, w: int =16):
        """
        Calculates the Inter-Symbol Interference (ISI) and signal-to-noise ratio for a given data vector,
        using a cosine roll-off transmit pulse.

        Parameters:
        -----------
        d : np.ndarray
            Input data vector (symbol sequence).
        r : float
            Roll-off factor for the transmit pulse shaping.
        L : int
            Number of symbol periods (must satisfy L >= 1/r, even).
        w : int, optional
            Number of samples per symbol interval (default is 16).

        Returns:
        --------
        None
            Prints ISI and SNR-related values to the console.
        """
        t, g = cosroll(r=r, w=w, L=L)                           # Generate time vector and transmit pulse

        _, x = datensig(g=g, w=w, d=d, A=True)                  # Generate transmitted signal
        _, xref = datensig(g=g, w=w, d=np.array([1]), A=True)   # Generate reference signal for single symbol

        delta: int = 1
        d_ref = d * xref[delta]             # Ideal reference signal, scaled to account for offset
        d_est: np.ndarray = x[delta::w]     # Sampled receive signal

        e: np.ndarray = (d_ref - d_est).__abs__()   # Absolute error
        s_n_isi: np.ndarray = (e / d_ref)**2        # Point-wise SNR through ISI

        s_n_isi_mean = np.mean(s_n_isi)

        e_max = e.max()

        print(f'r = {r}')
        print(f'Average SN (ISI) [dB] = {10 * np.log10(s_n_isi_mean)}')
        print(f'Max(ISI) = {e_max}\n')

    calculate_ISI(d=d, r=1, L=8)
    calculate_ISI(d=d, r=.5, L=16)
    calculate_ISI(d=d, r=0, L=256)


def exercise_three() -> None:
    """
    Solution of Tutorial 03, Exercise 03
    """
    def exercise_three_one(r: float, N: int) -> tuple[plt.Figure, plt.Axes]:
        """
        Solution of Tutorial 03, Exercise 3.1 for variable r and N

        Parameters:
        -----------
        r : float
            Roll-off factor.
            For r = 0, a truncated sinc function (cut at ±T·L/2) is used.

        N : int
            Number of data symbols contained in d.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object containing the eye diagram of the cosine roll-off impulse.
        ax : matplotlib.axes.Axes
            The Axes object of the plotted eye diagram.

        """
        d: np.ndarray = np.sign(np.random.randn(N))

        w = 16
        _, g = cosroll(r=r, w=w, L=256)
        _, x = datensig(g=g, w=w, d=d, A=True)

        fig, ax = auge(x, w)
        ax.set_title(f'Augendiagramm, r={r}')
        ax.set_xlabel(r'Normierte Zeit $t/T$')

        return fig, ax

    # r = 1
    fig1, ax1 = exercise_three_one(r=1, N=1000)
    ax1.annotate('', (-0.5, 0), (0.5, 0),
                 arrowprops=dict(arrowstyle='<->', linewidth=0.5, shrinkA=0, shrinkB=0))
    ax1.annotate('$\\frac{t_H}{T} = 1$', [0, -0.3], fontsize=4)
    ax1.annotate('', (1/16, 0), (1/16, 0.95),
                 arrowprops=dict(arrowstyle='<-', linewidth=0.5, shrinkA=0, shrinkB=0))
    ax1.annotate('$V = 0.95$', [0.08, 0.25], fontsize=4)
    plt.show()

    # r = .5
    fig2, ax2 = exercise_three_one(r=.5, N=1000)
    ax2.annotate('', (-0.4, 0), (0.4, 0),
                 arrowprops=dict(arrowstyle='<->', linewidth=0.5, shrinkA=0, shrinkB=0))
    ax2.annotate('$\\frac{t_H}{T} = 0.8$', [0, -0.3], fontsize=4)
    ax2.annotate('', (1 / 16, 0), (1 / 16, 0.87),
                 arrowprops=dict(arrowstyle='<-', linewidth=0.5, shrinkA=0, shrinkB=0))
    ax2.annotate('$V = 0.87$', [-0.05, 0.25], fontsize=4, bbox=dict(
        boxstyle='round,pad=0.2',
        facecolor='white',
        edgecolor='none',
        alpha=1
    ))
    plt.show()

    # r = 0
    fig3, ax3 = exercise_three_one(r=0, N=100000)
    ax3.annotate('', (-0.16, 0), (0.16, 0),
                 arrowprops=dict(arrowstyle='<->', linewidth=0.5, shrinkA=0, shrinkB=0))
    ax3.annotate('$\\frac{t_H}{T} = 0.32$', [-0.1, -0.3], fontsize=4)
    ax3.annotate('', (1 / 16, 0), (1 / 16, 0.63),
                 arrowprops=dict(arrowstyle='<-', linewidth=0.5, shrinkA=0, shrinkB=0))
    ax3.annotate('$V = 0.63$', [-0.05, 0.15], fontsize=4, bbox=dict(
        boxstyle='round,pad=0.2',
        facecolor='white',
        edgecolor='none',
        alpha=1
    ))
    plt.show()


exercise_one()
exercise_two()
exercise_three()
