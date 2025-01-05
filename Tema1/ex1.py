
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fft import dctn, idctn

def add_padding(image, block_size):
    rows, cols = image.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    padded_image = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
    return padded_image, rows, cols

def remove_padding(padded_image, original_rows, original_cols):
    return padded_image[:original_rows, :original_cols]

X = misc.ascent()
block_size = 8

Q_jpeg = [[
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]]
# adaug padding la imagine
X_padded, original_rows, original_cols = add_padding(X, block_size)

padded_rows, padded_cols = X_padded.shape

# imaginea JPEG reconstruita
X_jpeg_padded = np.zeros_like(X_padded, dtype=np.float32)
y_nnz, y_jpeg_nnz = 0, 0
for i in range(0, padded_rows, block_size):
    for j in range(0, padded_cols, block_size):
        block = X_padded[i:i + block_size, j:j + block_size]

        dct_block = dctn(block)

        quantized_block = Q_jpeg * np.round(dct_block / Q_jpeg)

        reconstructed_block = idctn(quantized_block)

        y_nnz += np.count_nonzero(dct_block)
        y_jpeg_nnz += np.count_nonzero(quantized_block)

        # salvam blocul în imaginea JPEG finala
        X_jpeg_padded[i:i + block_size, j:j + block_size] = reconstructed_block

# elimin padding-ul pentru a reveni la dimensiunea originala
X_jpeg = remove_padding(X_jpeg_padded, original_rows, original_cols)

print('Componente în frecvență:', y_nnz)
print('Componente în frecvență după cuantizare:', y_jpeg_nnz)


plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122)
plt.imshow(X_jpeg, cmap=plt.cm.gray)
plt.title('JPEG Compression')
plt.savefig('ex1.png')
plt.show()
