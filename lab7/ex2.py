import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

X = misc.face(gray=True)
rows, cols = X.shape

Y = np.fft.fft2(X)
Y_shifted = np.fft.fftshift(Y)

crow, ccol = rows // 2, cols // 2
radius = 50

Y_grid, X_grid = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
distance = np.sqrt((Y_grid - crow)**2 + (X_grid - ccol)**2)

mask = np.zeros((rows, cols))
mask[distance <= radius] = 1

Y_compressed = Y_shifted * mask

Y_compressed_shifted_back = np.fft.ifftshift(Y_compressed)
X_compressed = np.fft.ifft2(Y_compressed_shifted_back)
X_compressed = np.real(X_compressed)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(X, cmap='gray')
axs[1].imshow(np.log(1 + np.abs(Y_shifted)), cmap='gray')
axs[2].imshow(X_compressed, cmap='gray')

plt.savefig("ex2.pdf")
plt.show()
