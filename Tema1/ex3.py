
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fft import dctn, idctn


def rgb_to_ycbcr(image):
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    ycbcr = np.dot(image, transform_matrix.T)
    ycbcr[:, :, [1, 2]] += 128  # Adăugăm offset-ul pentru Cb și Cr
    return ycbcr


def ycbcr_to_rgb(image):
    inverse_transform_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    rgb = np.dot(image - [0, 128, 128], inverse_transform_matrix.T)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def calculate_mse(original, compressed):
    return np.square(np.subtract(original.flatten(), compressed.flatten())).mean()


def add_padding(image, block_size):
    rows, cols, channels = image.shape
    padded_rows = (rows + block_size - 1) // block_size * block_size
    padded_cols = (cols + block_size - 1) // block_size * block_size

    padded_image = np.zeros((padded_rows, padded_cols, channels), dtype=image.dtype)
    padded_image[:rows, :cols, :] = image
    return padded_image


def remove_padding(image, original_shape):
    return image[:original_shape[0], :original_shape[1], :]


Q_base = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

block_size = 8

X = misc.face()
rows, cols, channels = X.shape

X_ycbcr = rgb_to_ycbcr(X)

# adaugam padding imaginii
X_ycbcr_padded = add_padding(X_ycbcr, block_size)

#  dimensiunile imaginii cu padding
padded_rows, padded_cols, _ = X_ycbcr_padded.shape

# pragul MSE 
mse_threshold = 50

scaling_factor = 1.0
current_mse = float('-inf')
X_jpeg = np.zeros_like(X_ycbcr_padded, dtype=np.float32)
y_nnz, y_jpeg_nnz = 0, 0

while current_mse < mse_threshold:
    Q_jpeg = Q_base * scaling_factor

    for ch in range(channels):
        for i in range(0, padded_rows, block_size):
            for j in range(0, padded_cols, block_size):
                block = X_ycbcr_padded[i:i + block_size, j:j + block_size, ch]

                dct_block = dctn(block)

                quantized_block = np.round(dct_block / Q_jpeg) * Q_jpeg

                reconstructed_block = idctn(quantized_block)

                y_nnz = y_nnz + np.count_nonzero(dct_block)
                y_jpeg_nnz = y_jpeg_nnz + np.count_nonzero(quantized_block)

                X_jpeg[i:i + block_size, j:j + block_size, ch] = reconstructed_block

    X_jpeg_rgb = ycbcr_to_rgb(X_jpeg)

    X_jpeg_rgb = remove_padding(X_jpeg_rgb, X.shape)

    # calculam MSE
    current_mse = calculate_mse(X, X_jpeg_rgb)
    print(f"Current MSE: {current_mse} | Scaling Factor: {scaling_factor}")

    # crestem factorul de scalare pentru a ajusta compresia
    scaling_factor *= 8.1

print('Componente în frecvență:' + str(y_nnz) +
      '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X)
plt.title('Original')
plt.subplot(122)
plt.imshow(X_jpeg_rgb)
plt.title(f'JPEG Compression (MSE <= {mse_threshold})')
plt.savefig('ex3.png')
plt.show()
