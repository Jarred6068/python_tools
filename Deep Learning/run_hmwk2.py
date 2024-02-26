# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:28:15 2022

@author: Bruin
"""

import sys
sys.path.append('C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/')
import Homework_2_SGD_stu_final as linreg


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

## (1) Data preparation
df=pd.read_csv('C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/winequality-white.csv', sep = ';')
X = df.values[:, :11]
Y = df.values[:, 11]
print('Data shape:', 'X:', X.shape, 'Y:', Y.shape)

# data normalization
min_vals = np.min(X, axis = 0)
max_vals = np.max(X, axis = 0)
X1 = (X-min_vals)/(max_vals-min_vals)




## 2.1 Split the dataset into training (70%) and test (30%) sets. (5 points)
from sklearn.model_selection import train_test_split

## add your code here
#-----------------------
x_train,x_test,y_train,y_test = train_test_split(X1, Y, test_size = 0.3, train_size = 0.7)

print(x_train.shape, x_test.shape)
#---------------------------------

## 2.2 Model training using the training set and the GD function (5 points )
## add your code here
#-----------------------
w_star, loss_hist, w_hist = linreg.GD(x_train, y_train, lr = 1e-5)
#---------------------------------


mse_train = linreg.mse(w_star, x_train, y_train)
mae_train = linreg.mae(w_star, x_train, y_train)

print('training mse: {} and training mae:{}'.format(mse_train, mae_train))
#---------------------------------


## test error
## add your code here
#-----------------------

mse_test = linreg.mse(w_star, x_test, y_test)
mae_test = linreg.mae(w_star, x_test, y_test)

print('test mse: {} and test mae:{}'.format(mse_test, mae_test))
#---------------------------------


batch_size = 34
n_epochs = 50

#train model using SGD
w_star_SGD, w_hist_SGD, loss_hist_SGD = linreg.SGD(x_train, 
                                            y_train, 
                                            lr = 1e-2, 
                                            batch_size = batch_size, 
                                            epoch = n_epochs)