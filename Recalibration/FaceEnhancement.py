import cv2
import os

# 拉普拉斯算子
def sharpen_image(image_path):
    img = cv2.imread(image_path)
    # 对图像进行拉普拉斯边缘检测
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # 将边缘检测结果与原图像叠加，得到清晰化处理后的图像
    sharpened = cv2.addWeighted(img, 1.5, laplacian, -0.5, 0,dtype=cv2.CV_64F)
    return sharpened


def process_images(folder_path):
    # 遍历文件夹中的每一个文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        filepath = os.path.join(folder_path, filename)

        # 判断当前文件是否为图片文件
        if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 提升图像清晰度
            enhanced_image = sharpen_image(filepath)

            # 获取不带扩展名的文件名和路径名
            basename = os.path.splitext(filename)[0]
            dirname = os.path.dirname(filepath)

            # 新的图片名称
            enhanced_name = os.path.join(dirname, f'enhanced_{basename}.jpg')
            # 保存图像
            cv2.imwrite(enhanced_name, enhanced_image)

folder_path = "NJlocalface_test/Medium/faces/a(SBY)"
process_images(folder_path)