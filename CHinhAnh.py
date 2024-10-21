import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh từ tệp
image = cv2.imread('aaaaaaaa.png', cv2.IMREAD_GRAYSCALE)

# 1. Tạo ảnh âm tính
negative_image = 255 - image

# 2. Tăng độ tương phản
gamma = 1.2  # Giá trị gamma > 1 làm sáng, < 1 làm tối
contrast_image = np.array(255 * (image / 255) ** gamma, dtype='uint8')

# 3. Biến đổi log
log_image = np.array(255 * (np.log1p(image) / np.log1p(255)), dtype='uint8')

# 4. Cân bằng histogram
hist_eq_image = cv2.equalizeHist(image)

# 5. Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel theo trục x
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel theo trục y
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

# 6. Gaussian (LoG)
log_kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])

log_image_edge = cv2.filter2D(image, cv2.CV_64F, log_kernel)
log_image_edge = np.uint8(np.clip(log_image_edge, 0, 255))

# Hiển thị ảnh trước và sau xử lý
images = [image, negative_image, contrast_image, log_image, hist_eq_image, sobel_combined, log_image_edge]
titles = ['Original Image', 'Negative Image', 'Contrast Enhanced', 'Log Transformation', 'Histogram Equalization', 'Sobel Edge Detection', 'LoG Edge Detection']

plt.figure(figsize=(15, 10))

for i in range(7):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()

