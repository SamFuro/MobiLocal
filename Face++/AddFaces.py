import requests
from json import JSONDecoder
import cv2

http_url = "https://api-cn.faceplusplus.com/facepp/v3/faceset/addface"
key = "Q5r1XN_UULw52oRl9SN7O8exRjY5zaLf"
secret = "Gq9T_xFbSxvcX6OiY2_AB8YH4jxGBnj3"
data = {"api_key": key, "api_secret": secret, "outer_id": "NJUST","face_tokens":'04e51f54cb6b286b40eb48ffd15e3b18'}


response = requests.post(http_url, data=data)
print(response)
print(response.text)

# req_con = response.content.decode('utf-8')
# req_dict = JSONDecoder().decode(req_con)
#
# print(req_dict)
