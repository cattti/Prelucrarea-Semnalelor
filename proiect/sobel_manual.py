import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def load_image(path):
    return cv2.imread(path)


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convolve(image, kernel):
    #convoluția asupra imaginii folosind un kernel 3x3
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    output = np.zeros((h, w), dtype=np.float64)

    #aplic kernelul pe fiecare pixel
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            region = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            output[i, j] = np.sum(region * kernel)

    return output


def apply_sobel_operator_manual(image, threshold):
    # operatorii sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = grad_magnitude > threshold

    return grad_x, grad_y, grad_magnitude


def normalize_image(image):
    return np.uint8(255 * np.abs(image) / np.max(image))


def plot_images(images, titles):
    num_images = len(images)
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(2, 2, i + 1)
        cmap = 'gray' if len(images[i].shape) == 2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    image_path = 'paper4.jpeg'
    image = load_image(image_path)
    threshold = 150
    start_time = time.time()

    gray_image = convert_to_grayscale(image)
    grad_x, grad_y, grad_magnitude = apply_sobel_operator_manual(gray_image, threshold)

    grad_x_norm = normalize_image(grad_x)
    grad_y_norm = normalize_image(grad_y)
    grad_magnitude_norm = normalize_image(grad_magnitude)

    end_time = time.time()
    print(f"Edge detection completed in {end_time - start_time:.6f} seconds")

    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        grad_x_norm,
        grad_y_norm,
        grad_magnitude_norm
    ]
    titles = [
        "Original Image",
        "Gradient in X direction",
        "Gradient in Y direction",
        "Sobel Edge Detection"
    ]

    plot_images(images, titles)
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        cmap = 'gray' if len(images[i].shape) == 2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.savefig("triunghiSobelManual.jpeg", bbox_inches='tight')