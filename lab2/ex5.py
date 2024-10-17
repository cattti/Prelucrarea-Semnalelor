import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

fs = 44100
duration = 3

t = np.linspace(0, duration, fs*duration)

f1 = 200
x = np.sin(2 * np.pi * f1 * t)

f2 = 800
y = np.sin(2 * np.pi * f2 * t)

x_y = np.concatenate((x,y))

sd.play(x_y, fs)
sd.wait()

plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, duration * 2, len(x_y)), x_y)
plt.show()

#frecventa mai mare se aude mai inalta