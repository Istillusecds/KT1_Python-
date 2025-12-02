import numpy as np
import matplotlib.pyplot as plt





def exercise_one():

    # X-PLOT
    dt = 0.01
    t: np.ndarray = np.arange(0, 1 + dt, dt)

    # Y-PLOT Function of X
    s3 = np.abs(np.sin(2 * np.pi * t))
    s4 = s3 ** 2
##############################################################

    # X-PLOT
    #  I want 10 +1 Samples
    Ts3 = (1 - 0)/10  # nr of samples 10 + 1 on t-interval
    t3_samples = np.arange(0, 1 + Ts3, Ts3)
    # Y-PLOT
    s3_samples = np.abs(np.sin(2*np.pi* t3_samples))

    # X-PLOT
    # I want 4 +1 Samples
    Ts4 = (1 - 0)/4
    t4_samples = np.arange(0, 1 + Ts4, Ts4)
    # Y-PLOT
    s4_samples = np.abs(np.sin(2*np.pi* t4_samples))

###############################################################

    fig, ax = plt.subplots(2, 1)

    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,1])
    ax[0].set_xlabel("Zeit $t$ in sec")
    ax[0].set_ylabel("$s_3(t)$ / $s_3(k)$")
    ax[0].plot(t, s3)
    ax[0].stem(t3_samples, s3_samples, 'r:', basefmt=" ")

    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel("Zeit $t$ in sec")
    ax[1].set_ylabel("$s_4(t)$ / $s_4(k)$")
    ax[1].plot(t, s4)
    ax[1].stem(t4_samples, s4_samples, 'r:', basefmt=" ")

    fig.suptitle("Aufgabe 1.3")
    plt.show()


def exercise_two():

    A : np.ndarray = np.random.randint(1, 9, (2,2))
    B : np.ndarray = np.random.randint(1, 9, (2,2))

    print(A)
    print()
    print(B)
    print()

    C, D = matrix_multiplication(A, B)
    print(C, "\n")
    print(D)



def matrix_multiplication(A :np.ndarray, B :np.ndarray):

    """ function computes
         - classic matrix-matrix product C = A @ B
         - elementwise matrix product (Hadamard product) d_ij = a_ij * b_ij  if possible

    :param A  - np.ndarray
    :param B  - np.ndarray

    :return C - np.ndarray   C = A @ B
    :return D - np.ndarray
    """
    q = A.shape[1]
    r = B.shape[0]

    if q == r:
        C :np.ndarray = A @ B
    else: C = None

    if A.shape == B.shape:
        D :np.ndarray = A * B
    else: D = None

    return C, D

#exercise_one()
exercise_two()




