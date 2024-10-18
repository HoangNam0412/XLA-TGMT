import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh từ tệp
image = cv2.imread('aaaaaaaa.png', cv2.IMREAD_GRAYSCALE)

# 1. Tạo ảnh âm tính
negative_image = 255 - image

# 2. Tăng độ tương phản
# Sử dụng phương pháp tăng cường tương phản bằng cách điều chỉnh giá trị gamma
gamma = 1.2  # Giá trị gamma > 1 làm sáng, < 1 làm tối
contrast_image = np.array(255 * (image / 255) ** gamma, dtype='uint8')

# 3. Biến đổi log
log_image = np.array(255 * (np.log1p(image) / np.log1p(255)), dtype='uint8')

# 4. Cân bằng histogram
hist_eq_image = cv2.equalizeHist(image)

# Hiển thị ảnh trước và sau xử lý
images = [image, negative_image, contrast_image, log_image, hist_eq_image]
titles = ['Original Image', 'Negative Image', 'Contrast Enhanced', 'Log Transformation', 'Histogram Equalization']

plt.figure(figsize=(12, 8))

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
