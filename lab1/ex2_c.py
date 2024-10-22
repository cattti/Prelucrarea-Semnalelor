import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,5/240,100)
plt.plot(t,  (t%(1/240))/(1/240))
plt.savefig("ex2_c.pdf")
plt.show()