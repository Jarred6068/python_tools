# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:16:18 2022

@author: Bruin
"""

import os
import pandas as pd
import numpy as np
from tensorflow import keras as K
from tensorflow.keras.preprocessing import image
import tensorflow.keras.layers as layers 
from tensorflow.keras.layers import BatchNormalization
input_size = 500

# test_path = 'C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/F22_DL_HW4_train_test/testImgs'
# train_path = 'C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/F22_DL_HW4_train_test/train/'
# train_key = pd.read_csv('C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/F22_DL_HW4_train_test/train.csv')

test_path = '/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/testImgs'
train_path = '/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/train/'
train_key = pd.read_csv('/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/train.csv')


# load all training images and labels
train_img_folder = '/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/trainImgs'
train = pd.read_csv('/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/train.csv')
print(train)

trainImg_names = list(train['img name']) 

print('loading images ...')
X_train = []
for idx, name in enumerate(trainImg_names):
    img = image.load_img(os.path.join(train_img_folder, name +'.png'), 
                         color_mode = "rgb", 
                         target_size = (input_size, input_size))
    img = image.img_to_array(img)
    X_train.append(img)
    print(idx, name)
    







X_train2 = np.array(X_train).astype(np.float32)
y_train = np.array(train['tumor types'])

# convert class vectors to binary class matrices
y_train_onehot = K.utils.to_categorical(y_train)

print('loading finished ...')
print(X_train2.shape, y_train_onehot.shape)



#=============================Build-Model=======================================


# inputs = keras.Input(shape=(None, None, 3))
# processed = keras.layers.RandomCrop(width=32, height=32)(inputs)
# conv = keras.layers.Conv2D(filters=2, kernel_size=3)(processed)
# pooling = keras.layers.GlobalAveragePooling2D()(conv)
# feature = keras.layers.Dense(10)(pooling)

# full_model = keras.Model(inputs, feature)
# backbone = keras.Model(processed, conv)
# activations = keras.Model(conv, feature)

cnn_model = K.applications.Xception(
    weights='imagenet', 
    input_shape=(input_size, input_size, 3),
    include_top=False) 
 
# cnn_model = K.applications.InceptionResNetV2(
#     weights='imagenet', 
#     input_shape=(input_size, input_size, 3),
#     include_top=False) 

# cnn_model = K.applications.ResNet50(
#     weights='imagenet',  # Load weights pre-trained on ImageNet.
#     input_shape=(input_size, input_size, 3),
#     include_top=False)

# cnn_model = K.applications.DenseNet169(
#     weights='imagenet',  
#     input_shape=(input_size, input_size, 3),
#     include_top=False)


cnn_model.trainable = False


full_model = K.Sequential([layers.Input(shape=(input_size, input_size, 3)),
                           layers.Rescaling(scale=1./255),
                           layers.Normalization(axis=1),
                           layers.RandomRotation(factor = 0.3),
                           cnn_model,
                           layers.GlobalAveragePooling2D(),
                           layers.Flatten(),
                           layers.Dense(units = 128),
                           BatchNormalization(),
                           layers.Activation('relu'),
#                           layers.Dropout(.2),
                           layers.Dense(units = 64),
                           BatchNormalization(),
                           layers.Activation('relu'),
 #                          layers.Dropout(.2), 
                           layers.Dense(units = 2),
                           BatchNormalization(),
                           layers.Activation('softmax')])



full_model.compile(optimizer=K.optimizers.Adam(),
              loss=K.losses.CategoricalCrossentropy(),
              metrics='accuracy')

callback = K.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta = 1e-9)

full_model.summary()

full_model.fit(x=X_train2, y=y_train_onehot, epochs=100, batch_size = 64, validation_split=0.2,
               validation_batch_size = 32, callbacks=[callback])



#=============================Load-Test-Data=================================
import glob

os.chdir('/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/')

test_img_folder = 'testImgs'
testImg_names = glob.glob('testImgs/*.png')

print('loading test images ...')
X_test = []
for idx, name in enumerate(testImg_names):
    img = image.load_img(name, color_mode = "rgb", target_size = (input_size, input_size))
    img = image.img_to_array(img)
    X_test.append(img)
    print(idx, name)
    




    
X_test2 = np.array(X_test).astype(np.float32)
print('loading finished')


y_test_pred_onehot= full_model.predict(X_test2)

test_pred_class = np.argmax(y_test_pred_onehot, axis = 1) 

#===========================Fit-to-Test-Data=================================

df = pd.DataFrame({'name': list(testImg_names),
                   'pred': list(test_pred_class)})
df.to_csv('/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/test_pred.csv')












