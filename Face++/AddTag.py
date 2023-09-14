import requests
from json import JSONDecoder
import cv2

http_url = "https://api-cn.faceplusplus.com/facepp/v3/face/setuserid"
key = "Q5r1XN_UULw52oRl9SN7O8exRjY5zaLf"
secret = "Gq9T_xFbSxvcX6OiY2_AB8YH4jxGBnj3"

data = {"api_key": key, "api_secret": secret, 'face_token': '04e51f54cb6b286b40eb48ffd15e3b18', 'user_id': 'c'}

response = requests.post(http_url, data=data)
print(response)
print(response.text)


