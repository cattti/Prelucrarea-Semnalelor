import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np

#1
n = 100
x = np.random.rand(n)
fig, axs = plt.subplots(4)
for ax in axs:
    ax.plot(x)
    x = np.convolve(x, x)
plt.show()


#2
maxN = 5
ord_p = np.random.randint(1, maxN + 1)
ord_q = np.random.randint(1, maxN + 1)
coef_p = np.random.randint(1, 10, ord_p)
coef_q = np.random.randint(1, 10, ord_q)

p = np.poly1d(coef_p)
q = np.poly1d(coef_q)
r = np.convolve(p, q)
r = np.poly1d(r)
print(r)

p_fft = np.fft.fft(p.c, p.o + q.o + 1)
q_fft = np.fft.fft(q.c, p.o + q.o + 1)
r_fft = np.fft.ifft(p_fft * q_fft).real.round()
r_fft = np.poly1d(r_fft)
print(r_fft)