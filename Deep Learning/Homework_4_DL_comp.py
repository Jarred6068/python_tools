# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:08:29 2022

@author: Bruin
"""
import pandas as pd
import numpy as np
from matplotlib import image
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from tensorflow import keras as K
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import tensorflow.keras.layers as layers 
#import keras as K
#import keras.layers as layers
#from keras.preprocessing.image import ImageDataGenerator 

# test_path = 'C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/F22_DL_HW4_train_test/testImgs'
# train_path = 'C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/F22_DL_HW4_train_test/train/'
# train_key = pd.read_csv('C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/F22_DL_HW4_train_test/train.csv')

test_path = '/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/testImgs'
train_path = '/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/train/'
train_key = pd.read_csv('/mnt/ceph/jarredk/Assignments/Deep_Learning/F22_DL_HW4_train_test/train.csv')


onlyfiles = [ f for f in listdir(test_path) if isfile(join(test_path,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(test_path,onlyfiles[n]) )



aug_batch_size = 16
img_width, img_height = 231, 231

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=aug_batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_path, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=aug_batch_size,
    class_mode='binary',
    subset='validation') # set as validation data


#build model
model_cnn = K.Sequential()

# add convolution and pooling layers
model_cnn.add(layers.Conv2D(filters = 8, # number of filter
                            kernel_size=3, # 3*3
                            padding = 'same',
                            activation = 'relu',
                            input_shape = (img_width,img_height, 3),
                            strides = 2
                           ))
model_cnn.add(layers.MaxPooling2D(pool_size = 2, strides = 2))

model_cnn.add(layers.Conv2D(filters = 16,
                            kernel_size=5, # 5*5
                            padding = 'same',
                            activation = 'relu',
                            strides = 3
                            ))
model_cnn.add(layers.MaxPooling2D(pool_size = 2, strides = 2))

# add convolution and pooling layers
model_cnn.add(layers.Conv2D(filters = 32, # number of filter
                            kernel_size=3, # 3*3
                            padding = 'same',
                            activation = 'relu',
                            input_shape = (img_width,img_height, 3),
                            strides = 2
                           ))

model_cnn.add(layers.MaxPooling2D(pool_size = 2, strides = 2))

# model_cnn.add(layers.Conv2D(filters = 23, # number of filter
#                             kernel_size=5, # 3*3
#                             padding = 'same',
#                             activation = 'relu',
#                             input_shape = (img_width,img_height, 3),
#                             strides = 2
#                            ))

# model_cnn.add(layers.MaxPooling2D(pool_size = 2, strides = 2))
#flatten
model_cnn.add(layers.Flatten())

#FC layers
model_cnn.add(layers.Dense(units = 64, activation = 'relu'))
model_cnn.add(layers.Dense(units = 32, activation = 'relu'))
model_cnn.add(layers.Dense(units = 1, activation = 'sigmoid'))


model_cnn.compile(loss=K.losses.binary_crossentropy, optimizer=K.optimizers.Adam(), metrics=['accuracy'])
model_cnn.summary()

batch_size = 128
#model_cnn.fit(train_generator, batch_size = 128, epochs=10)
model_cnn.fit(
    train_generator,
    epochs=50,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size)





#============================================================================
#build model
model_cnn = K.Sequential()

# add convolution and pooling layers
model_cnn.add(layers.Conv2D(filters = 32, # number of filter
                            kernel_size=3, # 3*3
                            padding = 'same',
                            input_shape = (img_width,img_height, 3),
                            strides = 2
                           ))
model_cnn.add(BatchNormalization())
model_cnn.add(layers.Activation('relu'))
model_cnn.add(layers.GlobalMaxPooling2D(pool_size = 2, strides = 2))

model_cnn.add(layers.Conv2D(filters = 64,
                            kernel_size=5, # 5*5
                            padding = 'same',
                            strides = 3
                            ))
model_cnn.add(BatchNormalization())
model_cnn.add(layers.Activation('relu'))
model_cnn.add(layers.GlobalMaxPooling2D(pool_size = 2, strides = 2))

# add convolution and pooling layers
model_cnn.add(layers.Conv2D(filters = 64, #number of filter
                            kernel_size=5, # 3*3
                            padding = 'same',
                            strides = 3
                           ))
model_cnn.add(BatchNormalization())
model_cnn.add(layers.Activation('relu'))
model_cnn.add(layers.GlobalMaxPooling2D(pool_size = 2, strides = 2))

# model_cnn.add(layers.Conv2D(filters = 23, # number of filter
#                             kernel_size=5, # 3*3
#                             padding = 'same',
#                             activation = 'relu',
#                             input_shape = (img_width,img_height, 3),
#                             strides = 2
#                            ))

# model_cnn.add(layers.MaxPooling2D(pool_size = 2, strides = 2))
#flatten
model_cnn.add(layers.Flatten())

#FC layers
model_cnn.add(layers.Dense(units = 128))
model_cnn.add(BatchNormalization())
model_cnn.add(layers.Activation('relu'))
model_cnn.add(layers.Dense(units = 64))
model_cnn.add(BatchNormalization())
model_cnn.add(layers.Activation('relu'))
model_cnn.add(layers.Dense(units = 1))
model_cnn.add(BatchNormalization())
model_cnn.add(layers.Activation('sigmoid'))

model_cnn.compile(loss=K.losses.binary_crossentropy, optimizer=K.optimizers.Adam(), metrics=['accuracy'])
model_cnn.summary()

batch_size = 64
#model_cnn.fit(train_generator, batch_size = 128, epochs=10)
model_cnn.fit(
    train_generator,
    epochs=50,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size)
