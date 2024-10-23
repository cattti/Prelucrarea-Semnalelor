import matplotlib.pyplot as plt
import numpy as np


def distance_colors(z):
    distances = np.abs(z)
    normalized_distances = distances / np.max(distances)
    return plt.cm.viridis(normalized_distances)


f = 10
t = np.linspace(0,1,1000)
x = np.sin(2 * np.pi * f * t)

x_complex = x * np.exp(-2 * np.pi * 1j * t)

fig, axs = plt.subplots(1,2)
axs[0].plot(t, x)
colors = distance_colors(x_complex)
for i in range(len(x_complex) - 1):
    axs[1].plot(np.real(x_complex[i:i+2]), np.imag(x_complex[i:i+2]),
        color=colors[i],
        linewidth=3 )

axs[1].set_aspect('equal')
axs[1].grid(True)
plt.savefig('ex2fig1.png', format='png')
plt.savefig('ex2fig1.pdf', format='pdf')
plt.show()

###################################################################


omega = [1, f, 1.5 * f, 2 * f]
z = [x * np.exp(-2j * np.pi * o * t) for o in omega]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for ax, omega, z in zip(axs.flat, omega, z):
    colors = distance_colors(z)
    for i in range(len(z) - 1):
        ax.plot(np.real(z[i:i + 2]), np.imag(z[i:i + 2]), color=colors[i])

    ax.set_title(f'ω = {omega} Hz')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginar')
    ax.grid(True)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

plt.tight_layout()
plt.savefig('ex2fig2.png', format='png')
plt.savefig('ex2fig2.pdf', format='pdf')
plt.show()

