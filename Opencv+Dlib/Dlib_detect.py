import os
import dlib
from PIL import Image

# 文件夹路径
folder_path = 'images2/3target5m'

# 创建dlib窗口
# win = dlib.image_window()

# 运行之前，检查shape_predictor文件路径是否在相应的目录下
detector = dlib.get_frontal_face_detector()

# 遍历文件夹中的图片文件
for filename in os.listdir(folder_path):
    if filename.endswith(".JPG") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)

        # 加载图像
        img = dlib.load_rgb_image(image_path)

        # 检测脸部
        faces = detector(img)

        # 输出检测到的人脸数量
        print(f"Detected {len(faces)} face(s) in {filename}")

        # # 添加脸部叠加层
        # overlay = dlib.image_window()
        # for face in faces:
        #     overlay.add_overlay(face)
        #
        # # 将dlib图像类型转化为Pillow图像类型
        # pil_image = Image.fromarray(img)
        #
        # # 保存图片到文件中
        # output_filepath = os.path.join(folder_path, "Dlib_output_" + filename)
        # pil_image.save(output_filepath)
        #
        # # 等待窗口关闭
        # overlay.wait_until_closed()
        #
        # # 删除叠加层
        # del overlay