import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def load_image(path):
    """Load an image from a file."""
    return cv2.imread(path)


def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_sobel_operator(image, threshold):
    """Apply the Sobel operator to calculate gradients and magnitude."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = grad_magnitude > threshold
    return grad_x, grad_y, grad_magnitude


def normalize_image(image):
    """Normalize an image to the range 0-255."""
    return np.uint8(255 * np.abs(image) / np.max(image))


def plot_images(images, titles):
    """Plot multiple images with titles."""
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
    image_path = 'img.png'
    image = load_image(image_path)
    threshold = 150
    start_time = time.time()

    # Process the image
    gray_image = convert_to_grayscale(image)
    grad_x, grad_y, grad_magnitude = apply_sobel_operator(gray_image, threshold)

    # Normalize results
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
    # Save the figure
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        cmap = 'gray' if len(images[i].shape) == 2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.savefig("img.jpeg", bbox_inches='tight')