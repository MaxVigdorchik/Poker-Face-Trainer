import cv2
import requests

from .video_capture import MyVideoCapture
vids = MyVideoCapture()

def get_emotion_data():
    ret, rgb = vids.get_frame()
    result, jpg = cv2.imencode('.jpg', rgb)
    subscription_key = '1a58d3cda9554726b64dae6aba761774'
    jpg = jpg.tostring()

    emotion_recognition_url = "https://westeurope.api.cognitive.microsoft.com/face/v1.0/detect?returnFaceAttributes=emotion"

    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               "Content-Type": "application/octet-stream"}
    response = requests.post(
        emotion_recognition_url, headers=headers, data=jpg)
    response.raise_for_status()
    analysis = response.json()
    print(analysis)

get_emotion_data()
