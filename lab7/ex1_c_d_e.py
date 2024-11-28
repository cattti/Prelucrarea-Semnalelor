import numpy as np
import matplotlib.pyplot as plt


n1 = 24
n2 = 24


def f3(n1, n2):
    f3 = np.zeros((n1, n2))
    f3[0][5] = 1
    f3[0][n1-5] = 1
    return f3

def f4(n1, n2):
    f4 = np.zeros((n1, n2))
    f4[5][0] = 1
    f4[n1-5][0] = 1
    return f4


def f5(n1, n2):
    f5= np.zeros((n1, n2))
    f5[5][5] = 1
    f5[n1-5][n1-5] = 1
    return f5


f_3 = f3(n1, n2)
f_4 = f4(n1, n2)
f_5 = f5(n1, n2)

spectrum_3 = np.real(np.fft.ifft2(f3(n1, n2)))
spectrum_4 = np.real(np.fft.ifft2(f4(n1, n2)))
spectrum_5 = np.real(np.fft.ifft2(f5(n1, n2)))

fig, axs = plt.subplots(3, 2)

axs[0][0].imshow(f_3)
axs[1][0].imshow(f_4)
axs[2][0].imshow(f_5)

axs[0][1].imshow(spectrum_3)
axs[1][1].imshow(spectrum_4)
axs[2][1].imshow(spectrum_5)
plt.savefig("ex1_c_d_e.pdf")
plt.show()







