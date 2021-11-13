import asyncio
import io
import glob
import os
import sys
import time
import uuid
from azure.cognitiveservices.vision import face
from numpy.lib.type_check import imag
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pyautogui

key=""#Apply for azure password and terminal node by yourself
endpoint=""#Apply for azure password and terminal node by yourself
face_client = FaceClient(endpoint, CognitiveServicesCredentials(key))
face_landmarks= ['eyebrow_left_outer','eyebrow_left_inner', 'eye_left_outer', 'eye_left_top', 'eye_left_bottom',
'eye_left_inner', 'eyebrow_right_inner', 'eyebrow_right_outer', 'eye_right_inner', 'eye_right_top', 'eye_right_bottom', 'eye_right_outer']
face_attributes = ['age', 'gender', 'headPose', 'smile', 'facialHair', 'glasses', 'emotion']
with open('./azureface/testdata/2.jpg','rb+') as img:
    faces=face_client.face.detect_with_stream(img,return_face_landmarks=face_landmarks,detection_model="Detection_01",return_face_attributes=face_attributes)
print('Age: ', faces[0].face_attributes.age)
# image_path = os.path.join('./azureface/2.jpg')
# image_data = open(image_path, 'rb')
# subscription_key = key
# face_api_url = "https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect"
# headers = {'Content-Type': 'application/octet-stream',
# 'Ocp-Apim-Subscription-Key': subscription_key}
# params = {
# 'returnFaceId': 'true',
# 'returnFaceLandmarks': 'true'
# }
# response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
# response.raise_for_status()
# faces = response.json()
# print(faces)
if not faces:
    raise Exception('No face detected from image {}'.format("./azureface/testdata/1.jpg"))
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    #print(dir(rect))
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    
    return ((left, top), (right, bottom))

def getkeypoints(facesd):
    points=[]
    for face in facesd:
        landmarks=face.face_landmarks
        points.append(landmarks.eye_right_inner)
        points.append(landmarks.eye_right_outer)
        points.append(landmarks.eye_right_bottom)
        points.append(landmarks.eye_right_top)
        points.append(landmarks.eye_left_inner)
        points.append(landmarks.eye_left_outer)
        points.append(landmarks.eye_left_bottom)
        points.append(landmarks.eye_left_top)
    return points

def getiou(facesd,image):
    for face in facesd:
        y1=int(face.face_landmarks.eye_right_top.y)
        y2=int(face.face_landmarks.eye_right_bottom.y)
        x1=int(face.face_landmarks.eye_right_inner.x)
        x2=int(face.face_landmarks.eye_right_outer.x)
    print(x1,x2,y2,y1)
    return image[y1:y2,x1:x2]

def drawpoints(pointslist,image):
    for i in pointslist:
        cv2.circle(image, (int(i.x),int(i.y)),1,(0,0,255),4)

def modelpredict(mat):
    model=keras.models.load_model("./azureface/pretrain/simplebaseline.h5")
    model.summary()
    imgs=[]
    img=cv2.resize(mat,(160,160))
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img/255
    plt.imshow(img)
    plt.show()
    imgs.append(img)
    imgs=np.array(imgs)
    pre=model.predict(imgs)
    pre=np.squeeze(pre)
    plt.imshow(pre)
    plt.show()
    return pre

def solveheatmap(gray_res):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    closed1 = cv2.morphologyEx(gray_res, cv2.MORPH_CLOSE, kernel,iterations=1)    
    closed2 = cv2.morphologyEx(gray_res, cv2.MORPH_CLOSE, kernel,iterations=3)    
    opened1 = cv2.morphologyEx(gray_res, cv2.MORPH_OPEN, kernel,iterations=1)     
    opened2 = cv2.morphologyEx(gray_res, cv2.MORPH_OPEN, kernel,iterations=3)               
    plt.subplot(2,2,1) 
    plt.imshow(closed1)
    plt.subplot(2,2,2) 
    plt.imshow(closed2) #we choose closed2
    plt.subplot(2,2,3)
    plt.imshow(opened1)
    plt.subplot(2,2,4)
    plt.imshow(opened2)
    plt.show()
    return closed2

def getcoordinate(a):
    a=cv2.resize(a,(1600,900))
    index = np.unravel_index(a.argmax(), a.shape)
    print(index)
    pyautogui.moveTo(index[0],index[1],duration=0.25)


def main() :
    #image = Image.open('./azureface/2.jpg')
    image=cv2.imread('./azureface/testdata/2.jpg')
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image=cv2.resize(image,(1024,1024))
    #cv2.imwrite('./azureface/2.jpg',image)
    print('Drawing rectangle around face... see popup for results.')
    #draw = ImageDraw.Draw(image)
    iou=getiou(faces,image)
    plt.imshow(iou)
    plt.show()
    print(type(iou))
    for face in faces:
        #draw.rectangle(getRectangle(face), outline='red')
        cv2.rectangle(image,getRectangle(face)[0], getRectangle(face)[1], (0,255,0),1,4)
    drawpoints(getkeypoints(faces),image)
    #image.show()
    plt.imshow(image)
    plt.show()
    heatmap=modelpredict(iou)
    closed=solveheatmap(heatmap)
    getcoordinate(closed)

if __name__=="__main__":
    main()
