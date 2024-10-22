import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,3,2400)
plt.stem(t, np.sin(2 * np.pi * 800 * t))
plt.savefig("ex2_b.pdf")
plt.show()