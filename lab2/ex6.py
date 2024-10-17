import numpy as np
import matplotlib.pyplot as plt
import os
fs=1200

t = np.linspace(0, 1, 100)

f1 = fs/2
x = np.sin(2 * np.pi * f1 * t)

f2 = fs/4
y = np.sin(2 * np.pi * f2 * t)

f3 = 0
z = np.sin(2 * np.pi * f3 * t)

fig,axs = plt.subplots(3)
axs[0].plot(t,x)
axs[1].plot(t,y)
axs[2].plot(t,z)
plt.savefig(os.path.join("pdfs","ex6.pdf"))

plt.show()

# semnalul oscileaza mai rar odata cu reducerea frecventei