import matplotlib.pyplot as plt
import numpy as np
import os


t = np.linspace(0,1,1000)
x = np.sin(2 * np.pi * t)
y = np.cos(2 * np.pi * t - np.pi/2)

fig, axs = plt.subplots(2)
axs[0].plot(t,x)
axs[1].plot(t,y)
plt.savefig(os.path.join("pdfs", "ex1.pdf"))
plt.show()




