import numpy as np
import matplotlib.pyplot as plt
import os

fs = 1000
t = np.linspace(0, 1, 300)

signal = np.sin(2 * np.pi * fs * t)

# (a)
signal_1 = signal[::4]

# (b)
signal_2 = signal[1::4]


plt.figure(figsize=(10, 6))
fig, axs = plt.subplots(3)

axs[0].plot(t, signal)

t_1 = t[::4]
axs[1].plot(t_1, signal_1)


t_2 = t[1::4]
axs[2].plot(t_2, signal_2)
plt.savefig(os.path.join("pdfs","ex7.pdf"))
plt.show()

# al doilea semnal este mai putin detaliat decat primul. Al treilea semnal
# este la o diferenta de faza de al doilea semnal