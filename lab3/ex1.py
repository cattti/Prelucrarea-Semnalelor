import matplotlib.pyplot as plt
import numpy as np
import math


n = 8
F = np.zeros((n,n), dtype=np.complex_)

for l in range(n):
    for p in range(n):
        F[p][l] = math.e**(-2 * np.pi * 1j * l * p/n)*(1/np.sqrt(n))

real_F = F.real
imag_F = F.imag

fig, axes = plt.subplots( n, figsize=(15, 6))
col = [0,1,2,3,4,5,6,7]
for p in range(n):
    axes[p].plot(col, F[p].real)
    axes[p].plot(col, F[p].imag)

plt.savefig('ex1.png', format='png')
plt.savefig('ex1.pdf', format='pdf')
plt.show()
F_T = np.conjugate(F)
F_T = np.transpose(F_T)

FH_F = np.dot(F, F_T)

I_n = np.eye(n)
norm_difference = np.linalg.norm(FH_F - I_n)

print(norm_difference)