import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 0.5, int(0.5/0.003))
x = np.cos(520 * np.pi * t + np.pi/3)
y = np.cos(280 * np.pi * t - np.pi/3)
z = np.cos(120 * np.pi * t + np.pi/3)

fig, axs = plt.subplots(3)
axs[0].plot(t, x)
axs[1].plot(t, y)
axs[2].plot(t, z)

for ax in axs.flat:
    ax.set_xlim([0,0.5])
plt.savefig("ex1_a_b.pdf")
plt.show()




