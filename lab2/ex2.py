import matplotlib.pyplot as plt
import numpy as np
import os

fs = 1000
t = np.arange(0, 1, 1/fs)
f = 10
x = np.sin(2 * np.pi * f * t)
y = np.sin( 2 * np.pi * f * t + np.pi/2)
k = np.sin( 2 * np.pi * f * t + np.pi/3)
w = np.sin( 2 * np.pi * f * t + 3 * np.pi/4)

plt.plot(t,x)
plt.plot(t,y)
plt.plot(t,k)
plt.plot(t,w)
plt.savefig(os.path.join("pdfs", "ex2_1.pdf"))
plt.show()



z = np.random.normal(0, 1, len(t))

SNR = [0.1, 1, 10, 100 ]
norm_x = np.linalg.norm(x)
norm_z = np.linalg.norm(z)

for snr in SNR:
    gamma = np.sqrt(norm_x**2 / (snr * norm_z**2))
    x_noisy = x + gamma * z
    plt.plot(t, x_noisy)
    plt.show()
