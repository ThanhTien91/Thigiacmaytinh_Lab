import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread("car.jpg")

# Kiểm tra ảnh có đọc được không
if image is None:
    print("Không tìm thấy ảnh car.jpg")
    exit()

# 1. Chuyển ảnh sang HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 2. Lọc vùng biển số màu vàng
lower_yellow = np.array([15, 80, 80])
upper_yellow = np.array([40, 255, 255])

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# 3. Tìm contour
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Kiểm tra có tìm được contour không
if len(contours) == 0:
    print("Không tìm thấy vùng màu vàng!")
    exit()

# Tìm vùng màu vàng lớn nhất
contour = max(contours, key=cv2.contourArea)

# 4. Tìm hình chữ nhật bao quanh vùng biển số
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.float32(box)


# Hàm sắp xếp 4 điểm góc
def order_points(pts):
    result = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)

    # Góc trên trái
    result[0] = pts[np.argmin(s)]

    # Góc dưới phải
    result[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)

    # Góc trên phải
    result[1] = pts[np.argmin(d)]

    # Góc dưới trái
    result[3] = pts[np.argmax(d)]

    return result


# Sắp xếp 4 điểm
src = order_points(box)


# 5. Kích thước ảnh biển số sau khi làm thẳng
width = 400
height = 150

dst = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])


# 6. Perspective Transformation
M = cv2.getPerspectiveTransform(src, dst)

plate = cv2.warpPerspective(
    image,
    M,
    (width, height)
)


# 7. Hiển thị kết quả
cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Bien so da lam thang", plate)

cv2.waitKey(0)
cv2.destroyAllWindows()
