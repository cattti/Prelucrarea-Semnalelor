import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import sobel_manual


def gaussian_kernel(size=5, sigma=1.4):
    """
    Generează un kernel Gaussian.
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def non_maximum_suppression(grad_mag, grad_dir):
    """
    Aplică non-maximum suppression pentru a subția marginile.
    """
    h, w = grad_mag.shape
    output = np.zeros((h, w), dtype=np.float64)
    angle = grad_dir * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q, r = 255, 255

            #  vecinii in functie de directia gradientului
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q, r = grad_mag[i, j + 1], grad_mag[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q, r = grad_mag[i + 1, j - 1], grad_mag[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q, r = grad_mag[i + 1, j], grad_mag[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q, r = grad_mag[i - 1, j - 1], grad_mag[i + 1, j + 1]

            # Păstram doar maximele locale
            if grad_mag[i, j] >= q and grad_mag[i, j] >= r:
                output[i, j] = grad_mag[i, j]

    return output

def hysteresis_thresholding(image, low_threshold, high_threshold):
    """
    hysteresis thresholding pentru a elimina marginile slabe.
    """
    strong, weak = 255, 75
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))

    output = np.zeros_like(image, dtype=np.uint8)
    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak

    # Conecteaza marginile slabe la marginile puternice
    for i, j in zip(weak_i, weak_j):
        if np.any(output[i - 1:i + 2, j - 1:j + 2] == strong):
            output[i, j] = strong
        else:
            output[i, j] = 0

    return output

def canny_manual(image, low_threshold_ratio=0.03, high_threshold_ratio=0.09):

    kernel = gaussian_kernel()
    blurred = sobel_manual.convolve(image, kernel)

    grad_x, grad_y, grad_mag, grad_dir = sobel_manual.apply_sobel_operator_manual(blurred)

    suppressed = non_maximum_suppression(grad_mag, grad_dir)

    # hysteresis thresholding
    max_grad = np.max(suppressed)
    low_threshold = low_threshold_ratio * max_grad
    high_threshold = high_threshold_ratio * max_grad
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)

    return edges

if __name__ == "__main__":
    image_path = '/Users/dariatache/Desktop/PrelucrareSemnale/EdgeDetection/pythonProject/paper4.jpeg'
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        raise ValueError("Imaginea nu a putut fi încărcată.")

    start = time.time()
    edges = canny_manual(gray_image)
    end = time.time()

    print(f"Time: {end-start}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Imaginea originală")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Margini detectate (Canny manual)")
    plt.axis('off')

    plt.show()
    plt.savefig("paper4Canny_manual.jpeg")