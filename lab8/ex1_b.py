import numpy as np
import matplotlib.pyplot as plt

def autocorrelation(series):
    N = len(series)
    result = np.correlate(series, series, mode='full')
    return result[result.size // 2:] / (N - np.arange(N))

N = 1000
t = np.linspace(0, 1, N)

trend = 0.5 * t**2 + 0.3 * t + 0.1

freq1, freq2 = 10, 25
season = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.cos(2 * np.pi * freq2 * t)

noise = np.random.normal(0, 0.1, N)

time_series = trend + season + noise

auto_corr = autocorrelation(time_series)

plt.plot(auto_corr, color="blue")
plt.grid()
plt.savefig("ex1_b.pdf")
plt.show()