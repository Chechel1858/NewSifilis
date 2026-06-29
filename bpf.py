import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка
img = cv2.imread("noise_image.png", cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

# БПФ
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude = np.log1p(np.abs(fshift))

# Поиск пиков
mask = np.ones_like(fshift, dtype=np.uint8)
crow, ccol = rows // 2, cols // 2
threshold = magnitude.mean() + 3 * magnitude.std()
peak_coords = []

for y in range(rows):
    for x in range(cols):
        if np.sqrt((y - crow)**2 + (x - ccol)**2) < 30:
            continue
        if magnitude[y, x] > threshold:
            peak_coords.append((y, x))

# Подавление пиков
for y, x in peak_coords:
    y1, y2 = max(0, y - 5), min(rows, y + 5)
    x1, x2 = max(0, x - 5), min(cols, x + 5)
    mask[y1:y2, x1:x2] = 0

# Обратное БПФ
filtered = fshift * mask
img_result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

# Визуализация
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Исходное'), plt.axis('off')
plt.subplot(132), plt.imshow(magnitude, cmap='jet'), plt.title('Спектр'), plt.axis('off')
plt.subplot(133), plt.imshow(img_result, cmap='gray'), plt.title('Результат'), plt.axis('off')
plt.tight_layout(), plt.show()