import fftseq
import numpy as np
import matplotlib.pyplot as plt

for g in ("kt1.mplstyle", "../kt1.mplstyle", "../../kt1.mplstyle"):
    try:
        plt.style.use(g)
        break
    except OSError:
        continue
else:
    raise FileNotFoundError("kt1.mplstyle nicht gefunden in ., .. oder ../..")

def exercise_one():
    """Solution of exercise 1 - Dealing with random numbers"""

    '''Task 1.1'''
    # Initialization of random variables
    N: int = 1000000
    X: np.ndarray = np.zeros((3, N))
    X[0, :] = np.random.rand(N)  # X1
    X[1, :] = 2 + (3 - 2) * np.random.rand(N)  # X2
    X[2, :] = 10 + (20 - 10) * np.random.rand(N)  # X3
    print(f'mean of RV [X1, X2, X3]: {np.mean(X, axis=1)}')
    print(f'variance of RV [X1, X2, X3]: {np.std(X, axis=1)}')
    # create class containers for the histogram plot
    dx = 0.01
    bins1: np.ndarray = np.arange(-0.5, 1.1 + dx, dx)
    bins2: np.ndarray = np.arange(1.5, 3.1 + dx, dx)
    bins3: np.ndarray = np.arange(9.5, 20.1 + dx, dx)
    # centers of classes (for plotting)
    bin_centers1 = bins1 + 0.5 * dx
    bin_centers2 = bins2 + 0.5 * dx
    bin_centers3 = bins3 + 0.5 * dx
    # evaluate histogram data
    bincounts1 = np.histogram(X[0, :], bins1)[0]
    bincounts2 = np.histogram(X[1, :], bins2)[0]
    bincounts3 = np.histogram(X[2, :], bins3)[0]

    # plot section
    fig, ax = plt.subplots()
    ax.hist(bin_centers1[:-1], bins1, weights=bincounts1 / N,histtype='step', linewidth=0.6, label=rf'U[0,1]')
    ax.hist(bin_centers2[:-1], bins2, weights=bincounts2 / N, histtype='step', linewidth=0.6, label=rf'U[2,3]')
    ax.hist(bin_centers3[:-1], bins3, weights=bincounts3 / N, histtype='step', linewidth=0.6, label=rf'U[10,20]')
    ax.legend()
    ax.set_xlabel(rf'$x = x_k$')
    ax.set_ylabel(rf'Haeufigkeit von $x_k$')
    plt.show()

    '''Task 1.2'''
    print(f'')
    X_bin: np.ndarray = np.round(np.random.rand(N))
    print(f'p(X=1) = {np.count_nonzero(X_bin) / N} | '
          f'p(X=0) = {1 - np.count_nonzero(X_bin) / N}')

    X_bin_modified: np.ndarray = (np.random.rand(N) > 0.3).astype(int)
    print(f'p(X=1) = {np.count_nonzero(X_bin_modified) / N} | '
          f'p(X=0) = {1 - np.count_nonzero(X_bin_modified) / N}\n')

    '''Task 1.3'''
    X_normal: np.ndarray = np.zeros((3, N))
    X_normal[0, :] = np.random.randn(N)  # X1
    X_normal[1, :] = 10 * np.random.randn(N) + 5  # X2
    X_normal[2, :] = 5 * np.random.randn(N)  # X3

    bins: np.ndarray = np.arange(-5, 5 + 1)
    bincounts = np.histogram(X_normal[0, :], bins)[0]
    bins_mid = bins[:-1] + 0.5
    x: np.ndarray = np.arange(-5, 5 + 0.01, 0.01)
    f: np.ndarray = np.exp(-(x ** 2) / 2) * 1 / (np.sqrt(2 * np.pi))

    #############################################################################################
    fig, ax = plt.subplots()
    ax.stem(bins_mid, bincounts / N, basefmt=" ")
    ax.plot(x, f, color='red')
    ax.set_ylim(0, 0.4)
    ax.set_xlabel(rf'$x = x_k$')
    ax.set_ylabel(rf'relative Haeufigkeit von $x_k, f_x(x)$')
    plt.show()

    mean = np.mean(X_normal, axis=1)
    std = np.std(X_normal, axis=1)
    mu = '\u03BC'  # Unicode für mu
    sigma = '\u03C3'  # Unicode für sigma
    print(f'Random Variables Xi( {mu} | {sigma}): \n'
          f'------------------------------------------------\n'
          f'X1 = X1( {np.round(mean[0], 2)} | {np.round(std[0], 2)} )\n'
          f'------------------------------------------------\n'
          f'X2 = X2( {np.round(mean[1], 2)} | {np.round(std[1], 2)} )\n'
          f'------------------------------------------------\n'
          f'X3 = X3( {np.round(mean[2], 2)} | {np.round(std[2], 2)} )\n')

def exercise_two():
    """Solution of exercise 2 - fourier transform"""

    T: int = 4  # signal period in s
    fs: float = 1 / T  # largest signal frequency
    buffer: int = 10
    b: float = buffer * fs  # signal band with

    f_sample: float = 2 * b  # 1st Nyquist criterion
    dt: float = 1 / f_sample
    print(f'fs {fs}, f_sample {f_sample} dt {dt}')

    '''Task 2.1'''
    t1: np.ndarray = np.arange(-2-9*dt, -2 + dt/2, dt)
    t2: np.ndarray = np.arange(-2 + dt, -1 + dt/2, dt)
    t3: np.ndarray = np.arange(-1 + dt,  1 + dt/2, dt)
    t4: np.ndarray = np.arange( 1 + dt,  2 + dt/2, dt)
    t5: np.ndarray = np.arange( 2 + dt,  2 + 10.5*dt, dt)

    x1: np.ndarray = np.zeros(len(t1))
    x2: np.ndarray = np.linspace(0, 1, len(t2))
    x3: np.ndarray = np.ones(len(t3))
    x4: np.ndarray = np.linspace(1, 0, len(t4))
    x5: np.ndarray = np.zeros(len(t5))



    t: np.ndarray = np.hstack([t1, t2, t3, t4, t5])
    x: np.ndarray = np.hstack([x1, x2, x3, x4, x5])

    df_req: float = 0.01  # required frequency resolution
    X, x_zp, df = fftseq.fftseq(x, dt, df_req)
    X = dt * X
    X_shift = np.fft.fftshift(X)

    f: np.ndarray = (np.linspace(0, 1, len(X)) - 0.5) * (1 / dt)
    X_ana = 4 * (np.sinc(2 * f)) ** 2 - (np.sinc(f)) ** 2
    print('len(f) = ', len(f), 'len(X) = ', len(X))

    fig, ax = plt.subplots(2, 1, constrained_layout = True)
    fig.suptitle(r"Amplitudenspektrum von $x(t)$")
    ax[0].plot(f, np.abs(X_ana), label=r"$x(t)$", linewidth=0.7)
    ax[0].set_title(r"analytisch")
    ax[0].set_xlim(min(f), max(f))
    ax[0].grid(True)

    ax[1].plot(f, np.abs(X_shift), label=r"$\vert X(f)\vert$", linewidth=0.7)
    ax[1].set_xlabel('Frequenz f [Hz]')
    ax[1].set_xlim(min(f), max(f))
    ax[1].grid(True)
    ax[1].set_title(r"numerisch")


    def spectrum_ifft_signal(d: np.ndarray, w: float):
        """Compute the spectrum of a provided data vector by means of the IFFT.
        Apply zero padding, to obtain a higher resolved output signal

        Inputs
        ------
         d: np.ndarray
            binary data signal (frequency domain)
         w: float
            oversampling factor (used for zero-padding)

        Returns
        -------
        X: np.ndarray
            spectrum of the higher resolved data signal (frequency domain)
        """
        # X(f) --> x(t)
        x: np.ndarray = np.fft.ifft(d)
        # zero padding
        x_zp: np.ndarray = np.hstack([x, np.zeros((x.shape[1], int(w * d.size - x.size)))])
        # x(t) --> X(f)
        X = np.fft.fft(x_zp)

        return X

    N = 16
    w = 8
    data: np.ndarray = np.eye(N)
    D: np.ndarray = spectrum_ifft_signal(data, w)
    omega: np.ndarray = np.linspace(0, 1, D.shape[1])
    figure, axis = plt.subplots()
    for idx, hrfft in enumerate(D):
        axis.plot(omega, np.abs(hrfft), label=rf"$\vert X_{{{idx}}}(t)\vert$")
        axis.set_xlabel(r"normierte Frequenz $\Omega$")
        axis.set_ylabel('Amplitudenspektrum')
        axis.grid(True)
        axis.set_xlim(min(omega), max(omega))
        axis.set_ylim(min(np.abs(hrfft)), max(np.abs(hrfft)))
    plt.show()


#exercise_one()
exercise_two()