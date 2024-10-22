import matplotlib.pyplot as plt
import numpy as np


t = np.linspace(0, 1, 200)
x = np.cos(520 * np.pi * t + np.pi/3)
y = np.cos(280 * np.pi * t - np.pi/3)
z = np.cos(120 * np.pi * t + np.pi/3)

fig, axs = plt.subplots(3)
axs[0].stem(t, x)
axs[1].stem(t, y)
axs[2].stem(t, z)

for ax in axs.flat:
    ax.set_xlim([0,1])
plt.savefig("ex1_c.pdf")
plt.show()