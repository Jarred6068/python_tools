# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:48:04 2022

@author: Bruin
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


##(2) Assume a linear mode that y = w0*1 + w_1*x_1 +w_2*x_2+...+ w_11*x_11
def predict(X, w):
    '''
    X: input feature vectors:m*n
    w: weights
    
    return Y_hat
    '''
    # Prediction
    Y_hat = np.zeros((X.shape[0]))
    for idx, x in enumerate(X):          
        y_hat = w[0] + np.dot(w[1:].T, np.c_[x]) # linear model
        Y_hat[idx] = y_hat    
    return Y_hat

## (3) Loss function: L = 1/2 * sum(y_hat_i - y_i)^2
def loss(w, X, Y):
    '''
    w: weights
    X: input feature vectors
    Y: targets
    '''
    Y_hat = predict(X, w)
    loss = 1/2* np.sum(np.square(Y - Y_hat))
    
    return loss

# Optimization 1: Gradient Descent
def GD(X, Y, lr = 0.001, delta = 0.01, max_iter = 100):
    '''
    X: training data
    Y: training target
    lr: learning rate
    max_iter: the max iterations
    '''
    
    m = len(Y)
    b = np.reshape(Y, [Y.shape[0],1])
    w = np.random.rand(X.shape[1] + 1, 1)
    A = np.c_[np.ones((m, 1)), X]
    gradient = A.T.dot(np.dot(A, w)-b)
    
    loss_hist = np.zeros(max_iter) # history of loss
    w_hist = np.zeros((max_iter, w.shape[0])) # history of weight
    loss_w = 0
    i = 0                  
    while(np.linalg.norm(gradient) > delta) and (i < max_iter):
        w_hist[i,:] = w.T
        loss_w = loss(w, X, Y)
        print(i, 'loss:', loss_w)
        loss_hist[i] = loss_w
        
        w = w - lr*gradient        
        gradient = A.T.dot(np.dot(A, w)-b) # update the gradient using new w
        i = i + 1
        
    w_star = w  
    return w_star, loss_hist, w_hist



def mse(w,x,y):
    m = x.shape[0]
    x = np.c_[np.ones((m, 1)), x]
    y_pred = np.dot(x,w)
    MeanSquaredError = (1/m) * np.sum(np.square(y_pred.reshape(m,) - y))
    return MeanSquaredError
    
def mae(w,x,y):
    m = x.shape[0]
    x = np.c_[np.ones((m, 1)), x]
    y_pred = np.dot(x,w)
    MeanAbsoluteError = (1/m) * np.sum(np.absolute(y_pred.reshape(m,) - y))
    return MeanAbsoluteError


def SGD(X, Y, lr = 0.001, batch_size = 32, epoch = 100): 
    '''Implement the minibatch Gradient Desent approach
    
        X: training data
        Y: training target
        lr: learning rate
        batch_size: batch size
        epoch: number of max epoches
        
        return: w_star, w_hist, loss_hist
    '''
    m = len(Y)
    np.random.seed(9)
    w = np.random.rand(X.shape[1]+1, 1)    #(12,1) values in [0, 1)
    w_hist = [] # (epoch,12) 
    loss_hist = []            # (epoch,)
   
    
    ## add your code here
    #-----------------------
    for i in range(epoch):
        #(1) Shuffle data (X and Y) at the beginning of each epoch. (5 points)
        np.random.shuffle(X)
        np.random.shuffle(Y)
        
        #(2) go through all minibatches and update w. (30 points)
        for b in range(int(m/batch_size)): 
            # prepare the b mininath X_batch and Y_batch. 10 points
            idx = np.arange(b*32, b*32+32)
            m_batch = len(idx)
            X_batch = X[idx]
            Y_batch = Y[idx]
            
            #prepare A_batch and b_batch. 10 points
            b_batch = np.reshape(Y_batch, [Y_batch.shape[0],1])
            A_batch = np.c_[np.ones((m_batch, 1)), X_batch]
            
            
            #gradient calcualation and w update. 10 points
            gradient = A_batch.T.dot(np.dot(A_batch, w)-b_batch)
            w = w - lr*gradient
            #print(i, b_batch.shape, X_batch.shape, A_batch.shape)

            
            
        ## (3) Save the loss and current weight for each epoch. 5 points
        l = loss(w, X, Y)
        w_hist.append(w)
        loss_hist.append(l)
        print('Epoch', i, 'Loss: ', l)
        
        ##(4) Decay learning rate at the end of each epoch. 
        lr = lr * 0.9
    #---------------------------------
    
    w_star = w
    return w_star, w_hist, loss_hist  