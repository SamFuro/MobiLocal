import requests
from json import JSONDecoder
import cv2

http_url = "https://api-cn.faceplusplus.com/facepp/v3/faceset/create"
key = "Q5r1XN_UULw52oRl9SN7O8exRjY5zaLf"
secret = "Gq9T_xFbSxvcX6OiY2_AB8YH4jxGBnj3"

data = {"api_key": key, "api_secret": secret, 'display_name':'FaceSet2',"outer_id":"NJUST"}

response = requests.post(http_url, data=data)
print(response)
print(response.text)



