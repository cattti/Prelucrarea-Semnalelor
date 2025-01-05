import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fft import dctn, idctn

def add_padding(image, block_size):
    rows, cols, channels = image.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    padded_image = np.pad(image, ((0, pad_rows), (0, pad_cols), (0, 0)), mode='constant', constant_values=0)
    return padded_image, rows, cols

def remove_padding(padded_image, original_rows, original_cols):
    return padded_image[:original_rows, :original_cols, :]

def rgb_to_ycbcr(image):
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    ycbcr = np.dot(image, transform_matrix.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr

def ycbcr_to_rgb(image):
    inverse_transform_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    rgb = np.dot(image - [0, 128, 128], inverse_transform_matrix.T)
    return np.clip(rgb, 0, 255).astype(np.uint8)

Q_jpeg = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]
# Q_jpeg = [[17, 18, 24, 47, 99, 99, 99, 99],
#                           [18, 21, 26, 66, 99, 99, 99, 99],
#                           [24, 26, 56, 99, 99, 99, 99, 99],
#                           [47, 66, 99, 99, 99, 99, 99, 99],
#                           [99, 99, 99, 99, 99, 99, 99, 99],
#                           [99, 99, 99, 99, 99, 99, 99, 99],
#                           [99, 99, 99, 99, 99, 99, 99, 99],
#                           [99, 99, 99, 99, 99, 99, 99, 99]]

block_size = 8

X = misc.face()
print(X.shape)

# adaugam padding la imagine
X_padded, original_rows, original_cols = add_padding(X, block_size)
print(X_padded.shape)
# tranf imaginea din RGB în Y'CbCr
X_ycbcr_padded = rgb_to_ycbcr(X_padded)

X_jpeg_padded = np.zeros_like(X_ycbcr_padded, dtype=np.float32)
y_nnz, y_jpeg_nnz = 0, 0
print(X_jpeg_padded.shape)

# aplicam algoritmul JPEG pentru fiecare canal (Y, Cb, Cr)
rows_padded, cols_padded, channels = X_ycbcr_padded.shape
for ch in range(channels):
    for i in range(0, rows_padded, block_size):
        for j in range(0, cols_padded, block_size):
            block = X_ycbcr_padded[i:i + block_size, j:j + block_size, ch]

            dct_block = dctn(block)

            quantized_block = Q_jpeg * np.round(dct_block / Q_jpeg)

            reconstructed_block = idctn(quantized_block)

            y_nnz += np.count_nonzero(dct_block)
            y_jpeg_nnz += np.count_nonzero(quantized_block)

            # salvam blocul în imaginea JPEG finala
            X_jpeg_padded[i:i + block_size, j:j + block_size, ch] = reconstructed_block

# elimin padding-ul pentru a reveni la dimensiunea originala
X_jpeg = remove_padding(X_jpeg_padded, original_rows, original_cols)

# transf imaginea inapoi in RGB
X_jpeg_rgb = ycbcr_to_rgb(X_jpeg)

print('Componente în frecvență:' + str(y_nnz) +
      '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))


plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X)
plt.title('Original')
plt.subplot(122)
plt.imshow(X_jpeg_rgb)
plt.title('JPEG Compression')
plt.savefig('ex2.png')
plt.show()
