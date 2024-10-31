import numpy as np
import matplotlib.pyplot as plt

fs = 10
f0 = 16
t = np.linspace(0,1,1000, endpoint=False)
te = np.linspace(0,1, 10, endpoint=False)

x = np.sin( 2 * np.pi * 16 * t)
y = np.sin( 2 * np.pi * 46 * t)
z = np.sin( 2 * np.pi * 76 * t)

fig, axs = plt.subplots(3, 1)

xe = np.sin(2 * np.pi * 16 * te)
ye = np.sin(2 * np.pi * 46 * te)
ze = np.sin(2 * np.pi * 76 * te)

axs[0].plot(t, x)
axs[0].plot(te, xe, color="green", marker='o', linestyle="")
axs[0].set_xlim(0,0.5)

axs[1].plot(t, y)
axs[1].plot(te, ye, color="green", marker="o", linestyle="")
axs[1].set_xlim(0,0.5)

axs[2].plot(t, z)
axs[2].plot(te, ze, color="green", marker='o', linestyle="")
axs[2].set_xlim(0,0.5)

plt.show()
plt.savefig("ex2.png", format="png")
plt.savefig("ex2.pdf", format="pdf")
