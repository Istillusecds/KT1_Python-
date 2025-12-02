import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import Magic as magic

for g in ("kt1.mplstyle", "../kt1.mplstyle", "../../kt1.mplstyle"):
    try:
        plt.style.use(g)
        break
    except OSError:
        continue
else:
    raise FileNotFoundError("kt1.mplstyle nicht gefunden in ., .. oder ../..")


def exercise_one():

    # Input Set X
    dt = 0.01  # PLOT sample interval
    t: np.ndarray = np.arange(0, 1 + dt, dt)  #   plot time interval   (Lots of samples needed for good graph)

    # Output Multiple Functions y1, y2 as a function of x
    s3: np.ndarray = np.abs(np.sin(2 * np.pi * t))
    s4: np.ndarray = np.square(np.abs(np.sin(2 * np.pi * t)))

    # data sampling
    n1: np.ndarray = np.arange(0, 11)  # INDEX Interval sample points 0, 1, 2, ---, 10
    k1: np.ndarray = n1 / 10           # Normalized n1? Always from 0 to 1  ??????

    idx = (k1 / dt).astype(int)  # concrete sample index for function s3
                                 # astype(int) required since array is accessed via natural numbers
                                 #  1/ dt = N  Number of samples
    s3_sampled = s3[idx]


    n2: np.ndarray = np.arange(0, 5)
    k2: np.ndarray = n2 / 4    # normalized sampling index 0 to 1

    idx = (k2 / dt).astype(int)  #
    s4_sampled = s4[idx]

    ####################################################################################################
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, s3)
    ax[0].stem(k1, s3_sampled, 'b:', basefmt=" ")
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    ax[0].set_ylabel('$s_3(t)$ / $s_3(k)$')

    ax[1].plot(t, s4)
    ax[1].stem(k2, s4_sampled, 'b:', basefmt=" ")
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel('Zeit $t$ in sec')
    ax[1].set_ylabel('$s_4(t)$ / $s_4(k)$')

    fig.suptitle('Aufgabe 1.3')
    plt.show()


def matrix_multiplication(A: np.ndarray, B: np.ndarray)    -> Tuple[np.ndarray, np.ndarray]:
    """Function to compute
        - classic matrix-matrix product C = A @ B
        - element-wise matrix product (Hadamard product) d_ij = a_ij * b_ij
        if possible.

    :parameter A - np.ndarray
        Matrix of dimension (p, q)
    :parameter B - np.ndarray
        Matrix of dimension (r, s)

    :return C - np.ndarray
        Matrix of dimension (p, s) if q == r or None
    :return D - np.ndarray
        Matrix of dimension (p, q) == (r, s) or None
    """
    if A.shape[1] == B.shape[0]:
        C: np.ndarray = A @ B
    else:
        C = None

    if A.shape == B.shape:
        D = A * B
    else:
        D = None

    return C, D


def exercise_two():
    """Solution for exercise 2"""
    # Lambda function to create console output
    printline = lambda matrix: print(f'matrix with N = {matrix.shape[0]}: \n{matrix}\n'
                                     f'row sums: {matrix.sum(axis=0)}\n'
                                     f'col sums: {matrix.sum(axis=1)}\n'
                                     f'trace: {matrix.trace()}\n'
                                     f'magic? {magic.is_magic_matrix(matrix)}\n')

    '''Task 2.1'''
    A1: np.ndarray = magic.magic(8)
    printline(A1)

    '''Task 2.2'''
    C: np.ndarray = magic.magic(4)
    B1: np.ndarray = np.tile(C, (2, 2))
    printline(C)
    printline(B1)

    '''Task 2.3'''
    A2: np.ndarray = magic.magic(2)
    B2: np.ndarray = np.tile(A2, (2, 4))
    c: np.ndarray = A2.flatten()
    print("A2 = ", "\n")
    print(A2)
    print(f'B2 = {B2}\nc = {c}\n')

    # Test function matrix_multiplication with several matrices (function implementation above).
    A3: np.ndarray = np.ones((3, 3))
    A4: np.ndarray = np.linspace(1, 9, 9).reshape(3, 3)
    B3: np.ndarray = np.linspace(10, 20, 6).reshape(3, 2)

    C3_1, D3_1 = matrix_multiplication(A3, B3)
    print(f'C = {C3_1}, D = {D3_1}\n')
    C3_2, D3_2 = matrix_multiplication(A3, A4)
    print(f'C = {C3_2}, D = {D3_2}')


#exercise_one()
exercise_two()
