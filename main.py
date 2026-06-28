import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Загрузка изображения
# =========================
img = cv2.imread("noise_image.png", cv2.IMREAD_GRAYSCALE)

# =========================
# БПФ и центрирование
# =========================
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Логарифмический спектр
magnitude = np.log1p(np.abs(fshift))

# =========================
# Автоматический поиск пиков
# =========================
mask = np.ones_like(fshift, dtype=np.uint8)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Исключаем центральную область
center_radius = 30

# Порог для поиска ярких пиков
threshold = magnitude.mean() + 3 * magnitude.std()

peak_coords = []

for y in range(rows):
    for x in range(cols):

        # пропускаем центр спектра
        if np.sqrt((y - crow) ** 2 + (x - ccol) ** 2) < center_radius:
            continue

        if magnitude[y, x] > threshold:
            peak_coords.append((y, x))

# =========================
# Создание notch-фильтра
# =========================
notch_size = 5

for y, x in peak_coords:

    y1 = max(0, y - notch_size)
    y2 = min(rows, y + notch_size)

    x1 = max(0, x - notch_size)
    x2 = min(cols, x + notch_size)

    mask[y1:y2, x1:x2] = 0

# Применяем маску
filtered_shift = fshift * mask

# =========================
# Обратное БПФ
# =========================
f_ishift = np.fft.ifftshift(filtered_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# =========================
# Визуализация найденных пиков
# =========================
spectrum_rgb = cv2.cvtColor(
    cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8),
    cv2.COLOR_GRAY2RGB
)

for y, x in peak_coords:
    cv2.circle(spectrum_rgb, (x, y), 4, (255, 0, 0), 1)

# =========================
# Отображение результатов
# =========================
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.title("Исходное изображение")
plt.axis("off")

plt.subplot(132)
plt.imshow(spectrum_rgb)
plt.title("Спектр и найденные пики")
plt.axis("off")

plt.subplot(133)
plt.imshow(img_back, cmap="gray")
plt.title("После подавления шума")
plt.axis("off")

plt.tight_layout()
plt.show()