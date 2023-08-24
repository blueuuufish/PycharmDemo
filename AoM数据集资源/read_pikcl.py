import pickle
import cv2

# 打开并读取pickle文件
with open('VG_train.pkl', 'rb') as file:
    data = pickle.load(file)

for index, (key, value) in enumerate(data.items()):
    if index >= 10:
        break
    print(f"{key}: {value}")

# |-------------------------------------------------------------------------------|
# |                                                                               |
# |-------------------------------------------------------------------------------|
# 加载图片
image_path = "16_05_04_961.jpg"
image = cv2.imread(image_path)

# 从数据中获取裁剪的坐标
x, y, width, height = 25, 10, 134, 149
cropped_image = image[y: height, x: width]

# 保存裁剪后的图片
cv2.imwrite("cropped_image.jpg", cropped_image)

# 如果需要，可以显示裁剪后的图片
cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
