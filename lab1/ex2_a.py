import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 4, 1600 )
plt.stem(t, np.sin(2 * np.pi * 400 * t))
plt.savefig("ex2_a.pdf")
plt.show()