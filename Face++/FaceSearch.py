import os
import requests
from json import JSONDecoder


def process_images(folder_path):
    # 遍历文件夹中的每一个文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        filepath = os.path.join(folder_path, filename)

        # 判断当前文件是否为图片文件
        if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            detect_face(filepath)



def detect_face(image_path):
    try:
        http_url = "https://api-cn.faceplusplus.com/facepp/v3/search"
        key = "Q5r1XN_UULw52oRl9SN7O8exRjY5zaLf"
        secret = "Gq9T_xFbSxvcX6OiY2_AB8YH4jxGBnj3"

        data = {"api_key": key, "api_secret": secret, "outer_id": "wpx"}
        files = {"image_file": open(image_path, "rb")}
        response = requests.post(http_url, data=data, files=files)

        req_con = response.content.decode('utf-8')
        req_dict = JSONDecoder().decode(req_con)

        print(image_path, "有", req_dict["results"][0]["confidence"], "的概率认为是", req_dict["results"][0]["user_id"])
    except:
        print(image_path,'无法检测到人脸')


folder_path = "NJlocalface_test/Medium/faces/c(JJX)"
process_images(folder_path)
