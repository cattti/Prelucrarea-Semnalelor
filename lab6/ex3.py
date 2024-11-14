import numpy as np
import matplotlib.pyplot as plt

def hanning(W):
    return 0.5 * np.abs( 1 - np.cos( 2 * np.pi * np.arange(W)/W))

def rectangular(W):
    return np.ones(W)


f = 1000
w = 200
hanning_w = hanning(w)
rectangular_w = rectangular(w)

t = np.linspace ( 0,1, 1000)
x = np.sin(2 * np.pi * f * t)

x_rectangular = x[:w] * rectangular_w
x_hanning = x[:w] * hanning_w

fig, axs = plt.subplots(3)
axs[0].plot(t, x)
axs[1].plot(t[:w], x_rectangular)
axs[2].plot(t[:w], x_hanning)

plt.show()