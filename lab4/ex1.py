import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os

N = [128, 256, 512, 1024, 2048, 4096, 8192]
timpi_fisier = 'manual_fft.npy'
time_manual=[]
time_py=[]
t = np.linspace(0,1,10000)
x = np.sin( 2 * np.pi * 180 * t)

def fft_manual():
    for dft in N:
        start = time.time()
        F = np.zeros(dft,dtype=np.complex_)
        for l in range(dft):
            for p in range(dft):
                F[l] += x[p] + math.e ** (-2 * np.pi * 1j * l * p/ dft) * (1 / np.sqrt(dft))
        stop = time.time()
        time_manual.append(stop-start)
    np.save(timpi_fisier, time_manual)
    return time_manual

if os.path.exists(timpi_fisier):
    time_manual = list(np.load(timpi_fisier))
else:
    time_manual = fft_manual()


for dft in N:
    start2 = time.time()
    F = np.fft.fft(x, dft)
    stop2 = time.time()
    time_py.append(stop2 - start2)

print(time_manual, time_py)


plt.plot(N, time_manual)
plt.plot(N, time_py)
plt.yscale('log')
plt.savefig("ex1.png")
plt.savefig("ex1.pdf")

plt.show()
