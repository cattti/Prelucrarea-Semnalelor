import numpy as np
import matplotlib.pyplot as plt
import os

fs = 44100
t = np.linspace(0,1,100)
x = np.sin(2 * np.pi * 5 * t)
y = np.sign(np.sin(2 * np.pi * 300 * t))
sum = x + y
fig, axs = plt.subplots(3)
axs[0].plot(t,x)
axs[1].plot(t,y)
axs[2].plot(t,sum)
plt.savefig(os.path.join("pdfs","ex4.pdf"))
plt.show()