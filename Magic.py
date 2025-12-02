import numpy as np
from typing import Tuple


def odd_magic_square(N: int):
    """computes a magic matrix, i.e. a matrix whose entries are given by the integers
    1,...,N^2, whereby N is an odd number, based on the siamese method.
    based on: https://de.wikipedia.org/wiki/Magisches_Quadrat (February 10th, 2025)
    :parameter N: int
        dimension of the square magic matrix (odd)

    :returns magic: np.ndarray
        (N,N) numpy array, i.e. the magic matrix

    :raises ValueError
        if N is non-positive"""
    
    magic: np.ndarray = np.zeros((N, N))
    mid_id = int(np.floor(N / 2))
    row = 0
    col = mid_id
    magic[row, mid_id] = 1
    for number in range(2, N ** 2 + 1):
        if row - 1 < 0:
            next_row = N - 1
        else:
            next_row = row - 1
        if col + 1 > N - 1:
            next_col = 0
        else:
            next_col = col + 1
        if magic[next_row, next_col] == 0:
            row = next_row
            col = next_col
        else:
            row = (row + 1) % N  # Move down if the cell is already occupied
            col = col  # Stay in the same column
        magic[row, col] = number
    return magic


def doubly_even_magic_square(N: int):
    """computes a magic matrix, i.e. a matrix whose entries are given by the integers
        1,...,N^2, whereby N is an even number, which can be expressed as N = 4n +2,
         whereby n is a natural number. Algorithm is an implementation of the LUX-method.
         based on: https://de.wikipedia.org/wiki/LUX-Methode (February 10th, 2025)

        :parameter N: int
            dimension of the square magic matrix (even)

        :returns magic: np.ndarray
            (N,N) numpy array, i.e. the magic matrix"""

    n = (N - 2) // 4
    n_letter_col = N // 2
    lux: dict[str, np.ndarray] = {"L": np.asarray([[4, 1], [2, 3]]),
                                  "U": np.asarray([[1, 4], [2, 3]]),
                                  "X": np.asarray([[1, 4], [3, 2]])}

    l_mat: np.ndarray = np.tile("L", (n + 1, n_letter_col))
    u_mat: np.ndarray = np.tile("U", (1, n_letter_col))
    x_mat: np.ndarray = np.tile("X", (n - 1, n_letter_col))

    letter_mat: np.ndarray = np.vstack((l_mat, u_mat, x_mat))
    letter_mat[n + 1, n_letter_col // 2] = "L"
    letter_mat[n, n_letter_col // 2] = "U"

    factors: np.ndarray = odd_magic_square(2 * n + 1)
    factors = np.repeat(factors, 2, axis=0)
    factors = np.repeat(factors, 2, axis=1)
    magic: np.ndarray = np.zeros((N, N))
    count = 0
    for row_id in range(0, N, 2):
        for col_id in range(0, N, 2):
            letter: str = letter_mat[row_id // 2, col_id // 2]
            lux_mat: np.ndarray = lux.get(letter)
            factor_mat: np.ndarray = factors[row_id:row_id + 2, col_id: col_id + 2] - 1
            magic[row_id:row_id + 2, col_id: col_id + 2] = lux_mat + 4 * factor_mat
            count += 1
    return magic


def single_even_magic_square(N: int):
    """computes a magic matrix, i.e. a matrix whose entries are given by the integers
    1,...,N^2, whereby N is an even entry, which cannot be expressed as N = 4n +2,
    whereby n is a natural entry.
    based on: https://de.wikihow.com/Ein-magisches-Quadrat-lösen (February 10th, 2025)

    :parameter N: int
        dimension of the square magic matrix (even)

    :returns magic: np.ndarray
        (N,N) numpy array, i.e. the magic matrix"""

    magic: np.ndarray = np.zeros((N, N))
    corner_square_size: int = N // 4
    colors: dict[str, list[Tuple[int, int]]] = {"corner": [], "center": []}
    # identify corner tuples

    for row_idx in range(corner_square_size):
        for col_idx in range(corner_square_size):
            colors.get("corner").append((row_idx, col_idx))  # left corner top
            colors.get("corner").append((row_idx, N - 1 - col_idx))  # right corner top
            colors.get("corner").append((N - 1 - row_idx, col_idx))  # left corner bottom
            colors.get("corner").append((N - 1 - row_idx, N - 1 - col_idx))  # right corner bottom

    for row_idx in range(corner_square_size, N - corner_square_size):
        for col_idx in range(corner_square_size, N - corner_square_size):
            colors.get("center").append((row_idx, col_idx))  # inner square element

    entry: int = 1
    placed_numbers: set[int] = set()

    for row in range(N):
        for col in range(N):
            pos: Tuple[int, int] = (row, col)
            if colors.get("corner").__contains__(pos) or colors.get("center").__contains__(pos):
                magic[row, col] = entry
                placed_numbers.add(entry)
            entry += 1
    entry -= 1
    for row in range(N):
        for col in range(N):
            if magic[row, col] != 0:
                continue
            while placed_numbers.__contains__(entry):
                entry -= 1
            magic[row, col] = entry
            placed_numbers.add(entry)
    return magic


def magic(N: int):
    """computes a magic matrix, i.e. a matrix whose entries are given by the integers
    1,...,N^2. Each integer is only represented once and the row as well as the column sum of
    the matrix are identical

    :parameter N: int
        dimension of the square magic matrix

    :returns magic: np.ndarray
        (N,N) numpy array, i.e. the magic matrix

    :raises ValueError
        if N is non-positive"""

    if N <= 0:
        raise ValueError('Matrix needs to be of dimension 1 or greater!')
    if N == 1:
        magic: np.ndarray = np.asarray([1])
    elif N == 2:
        magic: np.ndarray = np.asarray([[1, 4], [2, 3]])
    else:
        if np.mod(N, 2) != 0:  # odd case
            magic: np.ndarray = odd_magic_square(N)
        elif np.mod((N - 2), 4) == 0:  # doubly even
            magic: np.ndarray = doubly_even_magic_square(N)
        else:
            magic: np.ndarray = single_even_magic_square(N)
    return magic



def is_magic_matrix(A:np.ndarray):
    """tests, whether the given matrix is a magic matrix or not

    :parameter A: np.ndarray
        the Matrix to check whether it is a magic matrix or not

    :return isMagic: bool
        True, if the matrix is magic, i.e. trace, row-, and column-sums are identical"""

    row_sum = np.sum(A, 0)
    col_sum = np.sum(A, 1)
    trace = np.trace(A)
    return (all(row_sum == col_sum) and row_sum[0] == trace)
