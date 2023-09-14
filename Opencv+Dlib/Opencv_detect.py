import os
import cv2

# 运行之前，检查cascade文件路径是否在相应的目录下
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 文件夹路径
folder_path = 'images2/3target5m'

# 遍历文件夹中的图片文件
for filename in os.listdir(folder_path):
    if filename.endswith(".JPG") or filename.endswith(".png"):  # 判断文件是否为图片文件
        image_path = os.path.join(folder_path, filename)

        # 读取图像
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图

        # 检测脸部
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(10, 10),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        print('Detected ', len(faces), " face in", filename)

        # 标记位置
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 255), 5)
            roi_gray = gray[y: y + h, x: x + w]
            roi_color = img[y: y + h, x: x + w]

        label = 'Result: Detected ' + str(len(faces)) + " faces !"
        cv2.putText(img, label, (10, 110),
                    cv2.FONT_HERSHEY_COMPLEX,
                    4.0, (255, 128, 0), 3)

        # 输出带标记的图像
        output_filepath = os.path.join(folder_path, "opencv_output_" + filename)
        cv2.imwrite(output_filepath, img)

        # # 显示图像并等待关闭窗口
        # cv2.namedWindow('img', 0)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()