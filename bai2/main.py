import cv2
import numpy as np

def auto_white_balance_gray_world(image):
    img_float = image.astype(np.float32)

    B, G, R = cv2.split(img_float)

    B_mean = np.mean(B)
    G_mean = np.mean(G)
    R_mean = np.mean(R)

    if B_mean == 0 or G_mean == 0 or R_mean == 0:
        return image

    Gray_avg = (B_mean + G_mean + R_mean) / 3.0

    B_new = B * (Gray_avg / B_mean)
    G_new = G * (Gray_avg / G_mean)
    R_new = R * (Gray_avg / R_mean)

    B_new = np.clip(B_new, 0, 255).astype(np.uint8)
    G_new = np.clip(G_new, 0, 255).astype(np.uint8)
    R_new = np.clip(R_new, 0, 255).astype(np.uint8)

    result = cv2.merge([B_new, G_new, R_new])

    return result


# Đọc ảnh
image = cv2.imread("sample.jpg")

print("Đang chạy chương trình...")

if image is None:
    print("KHÔNG ĐỌC ĐƯỢC ẢNH sample.jpg")
    exit()

print("Đọc ảnh thành công!")

# Auto White Balance
result = auto_white_balance_gray_world(image)

# Hiển thị
cv2.imshow("Original", image)
cv2.imshow("Auto White Balance", result)

cv2.waitKey(0)
cv2.destroyAllWindows()