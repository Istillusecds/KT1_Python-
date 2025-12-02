import numpy as np
import matplotlib.pyplot as plt
from Solution_2 import fftseq

def task_1_1():

    N = 1000
    x_rand = np.random.rand(N) + 2
    dx = 0.01
    bins = np.arange(2, 3 + dx , dx)

    #counts, edges = np.histogram(x_rand, bins=bins)

    mean = np.mean(x_rand)
    std = np.std(x_rand)

    ############################################################  Plotting
    fig, axis = plt.subplots()
    axis.set_xlim([2, 3])
    axis.set_ylim([0, 30])
    axis.set_xlabel("Bins")
    axis.set_ylabel("Counts")
    axis.hist(x_rand, bins=bins)
    fig.suptitle(f"mean = {mean} " + "     " + f"std = {std}")
    plt.show()

    print(x_rand)
    print()
    #print("counts = ", counts)
    print()
    print(bins)
    print(mean, std)


def task_1_2():

    def decide(data, zero_p):

        array = []
        p_zero_stat = 0

        for i in range(data.size):
            if data[i] <= zero_p:
                array.append(0)
                p_zero_stat += 1
            else:
                array.append(1)
        return p_zero_stat / data.size, array

    N :int = 10000
    x_rand = np.random.rand(N)
    p_zero, digital = decide(x_rand, 0.7)

    bins = np.linspace(0, 2, 3)
    mean_data = np.mean(x_rand)
    std_data = np.std(x_rand)

    counts, edges = np.histogram(digital, bins)
    max_count = np.max(counts)

    print(counts)
    print(p_zero)
    print(bins)
    print(digital)

    #  Plotting
    fig, axis = plt.subplots()
    axis.set_xlim([0,2])
    axis.set_ylim([0, max_count + 10])
    axis.hist(digital)
    plt.show()

def digital_stream(p_one):

    N :int = 10000
    data = np.random.rand(N) + p_one
    data = data.astype(int)

    ones = np.count_nonzero(data)
    print(ones/N)

def task_1_3():

    # Plot exact pdf curve of N(0,1)
    dx = 0.01  # x sample interval
    x     = np.arange(-5, 5 + dx, dx)  # x axis
    pdf_x = np.exp(-0.5 * (x ** 2)) / np.sqrt(2 * np.pi)  # pdf(x) axis


    # Plot histogram of realizations
    N = 1000000   # Number of realizations of RV X
    x_rand   = np.random.randn(N)
    x_bins   = np.arange(-5, 5 + 1, 1)   # x-axis [-5 -4 -3 -2 -1  0  1  2  3  4  5] WE DEFINE the edges
    counts, edges = np.histogram(x_rand, x_bins) # count axis,  edges is a copy of bins!!!

    # Plotting position of stems should be center of bin interval
    x_bins_mid = x_bins[:-1] + 0.5

    ############################################################################
    fig, axis = plt.subplots()
    axis.set_xlim([-5, 5])
    axis.set_ylim([0,  0.5])

    axis.stem(x_bins_mid, counts/N)
    plt.plot(x, pdf_x, color='red')
    plt.show()

def task_2_1():

    # Estimation of Signal Bandwidth
    T = 4 # Signal non-zero period in seconds
    fs = 1/T
    buffer = 10
    b = buffer * fs

    f_sample = 2*b
    dt = 1/f_sample
    print(f'fs {fs}, f_sample {f_sample} dt {dt}')

    t1 = np.arange(-2 -9*dt, -2 +dt/2, dt)
    t2 = np.arange(-2 + dt, -1 + dt/2, dt)
    t3 = np.arange(-1 + dt,  1 + dt/2, dt)
    t4 = np.arange( 1 + dt,  2 + dt/2, dt)
    t5 = np.arange( 2 + dt,  2 + 10.5*dt, dt)

    x1 = np.zeros(len(t1))
    x2 = t2 + 2
    x3 = 1 + 0 * t3
    x4 = 2 - t4
    x5 = np.zeros(len(t5))

    t = np.hstack([t1, t2, t3, t4, t5])
    x = np.hstack([x1, x2, x3, x4, x5])

    """
    t6 = np.arange(-9, -2, dt)
    t7 = np.arange(-2, 2, dt)
    t8 = np.arange(2, 9, dt)

    x6 = np.zeros(len(t6))
    x7 = np.ones(len(t7))
    x8 = np.zeros(len(t8))

    t_new = np.hstack([t6, t7, t8])
    x_new = np.hstack([x6, x7, x8])

    """

    df = 0.01
    X, m_zp, df = fftseq.fftseq(x, dt, df)
    X = dt * X
    X_shift = np.fft.fftshift(X)


    f = (np.linspace(0, 1, len(X)) -0.5)*(1/dt)
    print('len(f) = ', len(f), 'len(X) = ', len(X))
    X_ana = 4 * (np.sinc(2 * f)) ** 2 - (np.sinc(f)) ** 2
    #############################################################

    fig, axis = plt.subplots()
    axis.set_xlim([min(f), max(f)])
    axis.plot(f, abs(X_shift))
    axis.grid(True)

    plt.show()

def task_2_2(d :np.ndarray, w :float):

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
    x :np.ndarray = np.fft.ifft(d)
    x_zp = np.hstack( [x, np.zeros( (x.shape[1], int(w * d.size - x.size) ) ) ]    )
    X = np.fft.fft(x_zp)
    return X




