import numpy as np
import matplotlib.pyplot as plt


dt = 0.00001
t: np.ndarray = np.arange(0, 1 + dt, dt)
# Y-PLOT Function of X
x_rand = np.sin(2 * np.pi * t)

# bin_interval = 0.01
x_bins = np.arange(-1, 1 + 0.01, 0.01)

counts, edges = np.histogram(x_rand, x_bins)


x_bins_mid = edges[:-1] + 0.01/2  # bin_interval/2

############################
fig, ax = plt.subplots()
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.stem(x_bins_mid, counts/np.max(counts), markerfmt=' ')

print(np.std(x_rand))
# How to get rms from pdf alone???


plt.show()