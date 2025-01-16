import cv2
import matplotlib.pyplot as plt
import time

def apply_canny(img_path, low_threshold, high_threshold):
    # se incarca imaginea in matrice si se transforma
    # in grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # aplicarea unui filtru Gauss cu kernelsize 5x5 si
    # deviatia standard de 1.4
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # aplicare Canny cu threshold
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return img, edges


# Test the function
image_path = 'food.jpeg'
start = time.time()
original, canny_edges = apply_canny(image_path, 50, 150)
end = time.time()

print(f"Timpul de executie este {end-start}")
fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 1 rând, 2 coloane

ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
ax[0].set_title("Imaginea Originala:")
ax[0].axis('off')

ax[1].imshow(canny_edges, cmap='gray')
ax[1].set_title("Imaginea folosind Canny")
ax[1].axis('off')

plt.tight_layout()
plt.savefig("foodoriginalVScanny.jpeg")
plt.show()


