import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.linspace(0, 1, N)

trend = 0.5 * t**2 + 0.3 * t + 0.1

freq1, freq2 = 10, 25
season = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.cos(2 * np.pi * freq2 * t)
noise = np.random.normal(0, 0.1, N)

time_series = trend + season + noise

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t, trend, label="Trend", color="blue")
axs[0].legend()

axs[1].plot(t, season, label="Sezon", color="orange")
axs[1].legend()

axs[2].plot(t, noise, label="Zgomot", color="green")
axs[2].legend()

axs[3].plot(t, time_series, label="Serie de timp", color="red")
axs[3].legend()

plt.savefig("ex1_a.pdf")
plt.show()