import cv2
import os

def resize_image(image):
    height, width = image.shape[:2]
    max_dimension = max(height, width)

    if max_dimension > 4096:
        scale_factor = 4096.0 / max_dimension
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
    elif min(height, width) < 200:
        scale_factor = 200.0 / min(height, width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        resized_image = image
    return resized_image


def process_images(folder_path):
    # 遍历文件夹中的每一个文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        filepath = os.path.join(folder_path, filename)

        # 判断当前文件是否为图片文件
        if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 获取不带扩展名的文件名和路径名
            basename = os.path.splitext(filename)[0]
            dirname = os.path.dirname(filepath)

            # 读取图片
            image = cv2.imread(filepath)
            # 调整图片尺寸（假设resize_image()函数已实现）
            resized_image = resize_image(image)

            # 新的图片名称
            resized_name = os.path.join(dirname, f'resized_{basename}.jpg')
            # 保存图片到images中
            cv2.imwrite(resized_name, resized_image)

folder_path = "NJlocalface_test/Low/faces/c(JJX)"
process_images(folder_path)

