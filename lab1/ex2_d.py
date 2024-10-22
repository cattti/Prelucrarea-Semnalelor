import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-1,1, 300)
plt.plot(t, np.sign(np.sin(2 * np.pi * 300 * t)))
plt.savefig("ex2_d.pdf")
plt.show()