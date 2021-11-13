from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,GlobalAveragePooling2D,UpSampling2D,Input, BatchNormalization, LeakyReLU, concatenate,MaxPooling2D, AveragePooling2D, Conv2DTranspose,Convolution2D
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import cv2
import numpy as np
from tensorflow import keras
import numpy as np
import random
import os
from tensorflow.keras.models import save_model, load_model, Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
filename=os.listdir("./datasets/train/")
print(filename[4][-5])
imgs=[]
labels=[]
for i in filename:
    if(i[-4:]==".jpg"):
        img=cv2.imread("./datasets/train/"+i)
        print("./datasets/train/"+i)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=cv2.resize(img,(160,160))
        img=img/255
        # means=np.mean(img)
        # std=np.std(img)
        # img=(img-means)/std
        imgs.append(img)
        img=cv2.imread("./datasets/ground_truth/"+i[-5]+".jpg")
        print("./datasets/ground_truth/"+i[-5]+".jpg")
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=cv2.resize(img,(160,160))
        img=img/255
        # means=np.mean(img)
        # std=np.std(img)
        # img=(img-means)/std
        labels.append(img)
imgs=np.array(imgs)
labels=np.array(labels)
print(imgs.shape)
print(labels.shape)
model=Sequential()
model.add(keras.applications.ResNet50(include_top=False,weights=None,input_shape=(160,160,1)))
# model.add(keras.layers.UpSampling2D(size = (32, 32), interpolation = "bilinear",name = "upsamping_6"))
model.add(keras.layers.Conv2DTranspose(filters = 1024,kernel_size = (3, 3),strides = (2, 2),padding = "same",kernel_initializer = "he_uniform",name = "Conv2DTranspose_1"))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(keras.layers.Conv2DTranspose(filters = 512,kernel_size = (3, 3),strides = (2, 2),padding = "same",kernel_initializer = "he_uniform",name = "Conv2DTranspose_2"))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(keras.layers.Conv2DTranspose(filters = 256,kernel_size = (3, 3),strides = (2, 2),padding = "same",kernel_initializer = "he_uniform",name = "Conv2DTranspose_4"))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(keras.layers.Conv2DTranspose(filters = 128,kernel_size = (3, 3),strides = (4, 4),padding = "same",kernel_initializer = "he_uniform",name = "Conv2DTranspose_5"))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(Dropout(0.02))
model.add(Conv2D(1, kernel_size = (1, 1), activation = "sigmoid",padding = "same", name = "conv_1"))
model.compile(optimizer = "adam",loss = "mean_squared_error",metrics = ["acc"])
model.summary()
imgs = np.reshape(imgs, (len(imgs), 160,160,1))
labels = np.reshape(labels, (len(labels), 160,160,1))
his=model.fit(imgs,labels, epochs=80, batch_size=6,validation_split=0.1)
model.save("simplebaseline.h5")
e=len(his.history['loss'])
k=len(his.history['acc'])
plt.plot(range(e),his.history['loss'],label='Loss')
plt.plot(range(k),his.history['acc'],label='acc')
plt.legend()
plt.show()
loss_and_metrics = model.evaluate(imgs, labels, batch_size=8)
x = np.expand_dims(imgs[12], axis=0)
pre=model.predict(x)
pre=np.squeeze(pre)
pre.reshape(160,160,1)
print(pre.shape)
plt.imshow(pre)
plt.show()
y=labels[12]
y=np.squeeze(y)
y.reshape(1,160,160)
print(y.shape)
plt.imshow(y)
plt.show()