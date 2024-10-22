import matplotlib.pyplot as plt
import numpy as np

a = np.random.rand(128, 128)
plt.imshow(a)
plt.savefig("ex2_e.pdf")
plt.show()