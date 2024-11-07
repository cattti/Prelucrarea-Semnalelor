import numpy as np
import matplotlib.pyplot as plt

x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1)
x = x[:, 2]
N = len(x)
X = np.fft.fft(x)
X = abs(X/N)
X = X[:N//2]
Fs = 1/3600
f = Fs * np.linspace(0, N//2, N//2) / N

#1, g)
start_sample = 1224
end = 1224 + 30 * 24
x_new = x[start_sample:end]
time = f[start_sample:end]
plt.plot(time, x_new)
plt.title("Trafic pe o luna")
plt.savefig("ex1_g.pdf")
plt.show()