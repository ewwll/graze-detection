import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
imgs=[]
img=cv2.imread("./graze-detection/dataset/train/1_1.jpg")
model=keras.models.load_model("./graze-detection/pretrain/simplebaseline.h5")
model.summary()
img=cv2.resize(img,(160,160))
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img=img/255
plt.imshow(img)
plt.show()
imgs.append(img)
imgs=np.array(imgs)
pre=model.predict(imgs)
pre=np.squeeze(pre)
label=cv2.imread("./graze-detection/dataset/ground_truth/1.jpg")
label=cv2.resize(label,(160,160))
label=cv2.cvtColor(label,cv2.COLOR_RGB2GRAY)
label=label/255
plt.imshow(label)
plt.show()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
closed2 = cv2.morphologyEx(pre, cv2.MORPH_CLOSE, kernel,iterations=3) 
plt.imshow(closed2)
plt.show()
# # print(os.getcwd())
# # filname=os.listdir()
# # print(filname[20][-4::])
# # for k,i in enumerate(filname):
# #     if(i[-4::]==".txt"):
# #         os.rename(i,str(k)+"_graph.py")
# filname=os.listdir()
# print(filname[20][-9:])
# for k,i in enumerate(filname):
#     if(i[-9:]=="_graph.py"):
#         with open(i, 'r+') as file:
#             file.truncate(0)