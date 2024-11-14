import numpy as np
import matplotlib.pyplot as plt


x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1)
x = x[:, 2]
start = 100
final = start + 3*24
x = x[start:final]

w = [5, 9, 13, 17]
fig, axs = plt.subplots(5)

axs[0].plot(x)
i = 1
for wind in w:
    axs[i].plot((np.convolve(x, np.ones(wind),"valid")/wind))
    i += 1
plt.show()