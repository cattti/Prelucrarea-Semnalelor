import numpy as np
import matplotlib.pyplot as plt


#1, a)
# se esantioneaza o data pe ora, frecventa se calculeaza pe secunda, deci 60 de secunde cu 60 de minute
# fs = 1/3600

#1, b)
# 18288/24 = 762 zile

#1, c)
# pentru a nu suferi aliere, frec max = fs/2

#1, d)
x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1)
x = x[:, 2] #pastrez doar valorile din ultima coloana
N = len(x)
X = np.fft.fft(x)
X = abs(X/N)
X = X[:N//2]
Fs = 1/3600
f = Fs * np.linspace(0, N//2, N//2) / N


#1, e)
componenta_continua = X[0]
print(componenta_continua)

fig, axs = plt.subplots(2)
axs[0].plot(f, X)

x = x - np.mean(x)
X = np.fft.fft(x)
X = abs(X/N)
X = X[:N//2]

cmp = X[0]
print(cmp)

axs[1].plot(f, X)
plt.savefig("ex1.pdf")
plt.show()



#1, f)

#cele mai mari 4 amplitudini și indexii lor
top_4_indices = np.argsort(X)[-4:]

#cele mai mari 4 amplitudini și frecvențele lor
top_4_amplitudes = X[top_4_indices]
top_4_freqs = f[top_4_indices]
for i in range(4):
    print(f"Amplitudine: {top_4_amplitudes[i]}, Frecvență: {top_4_freqs[i]} Hz")


