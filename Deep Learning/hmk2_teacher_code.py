# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:42:04 2022

@author: Bruin
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

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




from sklearn.model_selection import KFold

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
    Y = Y[:,None] #to ensure input vector for y is 2D
    np.random.seed(9)
    w = np.random.rand(X.shape[1]+1, 1)    #(12,1) values in [0, 1)
    #w_hist = np.zeros((epoch, w.shape[0])) # (epoch,12) 
    w_hist = []
    loss_hist = []            # (epoch,)
    #batch_weights = [w]
    
    #preallocate the epoch training and testing losses
    epoch_train_loss = []
    #epoch_valid_loss = []
    ## add your code here
    #-----------------------
    for i in range(epoch):
        #(1) Shuffle data (X and Y) at the beginning of each epoch. (5 points)
        np.random.shuffle(X)
        np.random.shuffle(Y)
        kf = KFold(n_splits=int(X.shape[0]/batch_size))
        #preallocate the batch training and validation loss
        batch_train_loss=[]
        #batch_valid_loss=[]

        
        #(2) go through all minibatches and update w. (30 points)
        for valid_index, batch_index in kf.split(X):
            
            m_batch = len(batch_index)
            # prepare the X_batch and Y_batch. 10 points
            X_batch_train, Y_batch_train = X[batch_index], Y[batch_index]
            #X_batch_valid, Y_batch_valid = X[valid_index], Y[valid_index]
                
            #prepare A_batch and b_batch. 10 points
            A_batch_train = np.c_[np.ones((X_batch_train.shape[0], 1)), X_batch_train] #design matrix for X batch
            #A_batch_valid = np.c_[np.ones((batch_size, 1)), X_batch_train] #design matrix for everything not in X batch

            #gradient calcualation and w update. 10 points
            gradient = A_batch_train.T.dot(np.dot(A_batch_train, w)-Y_batch_train)
            #print(i, Y_batch_train, X_batch_train.shape, A_batch_train.shape)
                
            #calculate new weights
            w = w - lr*gradient
            #store batch train/test loss
            batch_train_loss.append(loss(w, X_batch_train, Y_batch_train.reshape(m_batch,)))
            #batch_valid_loss.append(loss(weights_new, X_batch_valid, Y_batch_valid))
                
            
            
        ## (3) Save the loss and current weight for each epoch. 5 points
        epoch_train_loss.append(loss(w, X, Y.reshape(m, )))
        print('Epoch ',i+1, ' train loss: ', epoch_train_loss[-1], ', best batch loss: ', min(batch_train_loss))
        #store the best epoch training and testing loss
        
        #epoch_test_loss.append(min(batch_test_loss))
        #print(batch_weights)
        #print(weights_new)
        #update the weight history
        #print(batch_weights[-1].tolist())
        w_hist.append(w)
        #print(i, loss_hist[i])
        
        ##(4) Decay learning rate at the end of each epoch. 
        lr = lr * 0.9
    #---------------------------------
    
    w_star = w_hist[-1]
    loss_hist = epoch_train_loss
    return w_star, w_hist, loss_hist  