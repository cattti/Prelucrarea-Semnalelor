import numpy as np
import matplotlib.pyplot as plt
import cv2
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
block_size = 8

input_video_path = 'video.MOV'
output_video_path = 'output_compressed_video.mp4'

cap = cv2.VideoCapture(input_video_path)

# informațiile despre video
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# procesam fiecare cadru
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # adaug padding la cadru
    frame_padded, original_rows, original_cols = add_padding(frame, block_size)

    # RGB in Y'CbCr
    frame_ycbcr_padded = rgb_to_ycbcr(frame_padded)

    # imag JPEG comprimata
    frame_jpeg_padded = np.zeros_like(frame_ycbcr_padded, dtype=np.float32)
    y_nnz, y_jpeg_nnz = 0, 0

    # aplicam algoritmul JPEG pentru fiecare canal (Y, Cb, Cr)
    rows_padded, cols_padded, channels = frame_ycbcr_padded.shape
    for ch in range(channels):
        for i in range(0, rows_padded, block_size):
            for j in range(0, cols_padded, block_size):
                block = frame_ycbcr_padded[i:i + block_size, j:j + block_size, ch]

                dct_block = dctn(block)

                quantized_block = Q_jpeg * np.round(dct_block / Q_jpeg)

                reconstructed_block = idctn(quantized_block)

                y_nnz += np.count_nonzero(dct_block)
                y_jpeg_nnz += np.count_nonzero(quantized_block)

                frame_jpeg_padded[i:i + block_size, j:j + block_size, ch] = reconstructed_block

    # eliminam padding-ul
    frame_jpeg = remove_padding(frame_jpeg_padded, original_rows, original_cols)

    # transformam imaginea inapoi in RGB
    frame_jpeg_rgb = ycbcr_to_rgb(frame_jpeg)

    # scriem cadrul comprimat in fisierul de iesire
    out.write(frame_jpeg_rgb)

    print(f"Procesat cadrul {cap.get(cv2.CAP_PROP_POS_FRAMES)}")

cap.release()
out.release()

print("Compresia video-ului a fost finalizată!")

cap = cv2.VideoCapture('output_compressed_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()