# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:52:03 2022

@author: Bruin
"""


import numpy as np
import math as m
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

class linreg:
    """implementation of Multiple Regression"""
    """Uses GD to determine est. coefficients:"""
    
    #model initialization
    def __init__(self, X, Y, lr = 0.001, conv_crit = 0.01, max_iter = 1000, 
                 train_test_split = [0.7, 0.3], optimizer=['MSE', 'LSE'],
                 normalize = True, verbose = False, method = ['GD', 'SGD'],
                 epochs = 10, batchsize = 32, decay = 0.9):
        """ initialization of data attributes in "data" """
        
        """ X should be an np.array where each row represents the
            values of the explanatory variable(s) for the i{th} 
            observation
        """
        if normalize == True:
            self.data = (X - np.mean(X, 0))/np.std(X, 0)    
        else:
            self.data = X
        self.target = Y[:, None]
        self.num_obs = X.shape[0]
        self.num_features = X.shape[1]
        self.conv_crit = conv_crit
        self.max_iter = max_iter
        self.learning_rate = lr
        self.percent_train = train_test_split[0]
        self.percent_test = train_test_split[1]
        self.optimizer = optimizer
        self.verbose = verbose
        self.batchsize = batchsize
        self.epochs = epochs
        self.decay = decay
        self.method = method
        
    #obtain the training and testing splits of the data   
    def split_data(self):
        train_x, test_x, train_y, test_y = tts(self.data, self.target, 
                          train_size = self.percent_train,
                          test_size = self.percent_test)
        
        self.train_data = train_x
        self.test_data = test_x
        self.train_target = train_y
        self.test_target = test_y
        



    #function to fit the linear model of Y ~ BX
    def fit(self):
        
        if self.method == 'GD':
            weights, train_loss, test_loss, change_loss = self.GD()
        if self.method == 'SGD':
            weights, train_loss, test_loss, change_loss = self.SGD()
        
        self.param_history = weights
        self.train_loss_history = train_loss
        self.test_loss_history = test_loss
        self.change_loss_history = change_loss
        self.final_weights = self.param_history[-1]
        self.preds = self.predict(weights = None, which = 'all')
        self.final_loss = self.loss(y_pred = self.preds, n = self.num_obs,
                                    which='all')
        
        if self.verbose == True:
            print('final_loss: ', self.final_loss)
        
    
    #function to execture the gradient descent estimation of model weights
    def GD(self):
        t=+1
        train_loss_vec = [1]
        change_loss = 1
        test_loss_vec = [1]
        weights = [np.random.rand(self.num_features+1, 1)]
            
        while (change_loss > self.conv_crit) and (t < self.max_iter):
            #calculate the loss with given parameters
            d_t = self.GD_get_grad(weights[t-1])
            
            weights_new = weights[t-1] - self.learning_rate*d_t
            
            Y_hat_train = self.predict(weights_new, which = 'train')
            
            Y_hat_test = self.predict(weights_new, which='test')
            
            train_loss = self.loss(n = self.train_data.shape[0], 
                                   y_pred = Y_hat_train, 
                                   which = 'train')
            
            test_loss = self.loss(y_pred=Y_hat_test, 
                                  n = self.test_data.shape[0],
                                  which='test')
            
            change_loss = abs(train_loss_vec[t-1] - train_loss)
            if self.verbose == True:
                print('iteration_',t, ' train_loss: ', train_loss, ' test_loss:',
                      test_loss)
            #update parameters
            weights.append(weights_new)
            train_loss_vec.append(train_loss)
            test_loss_vec.append(test_loss)
            t+=1
            
            
        self.its = np.arange(t-1)
        return weights, train_loss_vec, test_loss_vec
    
    
    
    def SGD(self):
        t=+1
        train_loss_history = [0]
        change_loss_history = [1]
        test_loss_history = [0]
        weights_history = []
        weights = [np.random.rand(self.num_features+1, 1)]
            
        while (change_loss_history[t-1] > self.conv_crit) and (t < self.max_iter):
            
            
            epoch_train_loss=[]
            epoch_test_loss=[]
            
            for i in np.arange(self.epochs):
                batch_weights = [weights[-1]]
                np.random.shuffle(self.train_data)
                np.random.shuffle(self.train_target)
                
                kf = KFold(n_splits=int(self.train_data.shape[0]/self.batchsize))
                train_batch_loss=[]
                test_batch_loss=[]
                #batch_change_loss=[]
                # if self.verbose == True:
                #     print("Epoch ",i+1, " number of splits = ", kf.get_n_splits(self.train_data))
                for test_index, batch_index in kf.split(self.train_data):
                    
                    x_train_b, x_test_b = self.train_data[batch_index], self.train_data[test_index]
                    y_train_b, y_test_b = self.train_target[batch_index], self.train_target[test_index]
                    
                    #calculate the loss with given parameters
                    d_t = self.SGD_get_grad(weights=batch_weights[-1], 
                                            X = x_train_b, 
                                            y = y_train_b)
                    
                    weights_new = batch_weights[-1] - self.learning_rate*d_t 
                    
                    Y_hat_train_b = self.predict(weights = weights_new, 
                                               which = 'minibatch',
                                               X = x_train_b)
                    Y_hat_test_b = self.predict(weights = weights_new, 
                                              which = 'minibatch',
                                              X = x_test_b)
                    
                    train_batch_loss.append(self.loss(n = x_train_b.shape[0], 
                                                      y_pred = Y_hat_train_b, 
                                                      which = 'minibatch',
                                                      y_true = y_train_b))
                    
                    test_batch_loss.append(self.loss(n = x_test_b.shape[0],
                                                     y_pred = Y_hat_test_b,
                                                     which='minibatch',
                                                     y_true = y_test_b))
                    
                    batch_weights.append(weights_new)
                    
                if self.verbose == True:
                   print('Epoch ',i+1, ' best train loss: ', min(train_batch_loss), ' best test loss:',
                              min(test_batch_loss))
            
            
                #update parameters
                weights[-1] = batch_weights[-1]
                epoch_train_loss.append(min(train_batch_loss))
                epoch_test_loss.append(min(test_batch_loss))
            
            
            train_loss_history.append(min(epoch_train_loss))
            test_loss_history.append(min(epoch_test_loss))
                
            change_loss_history.append(abs(train_loss_history[t-1] - train_loss_history[t]))
            
            weights_history.append(weights[-1])
            self.learning_rate = self.learning_rate*self.decay
            t+=1
            
            
        self.its = np.arange(t-1)
        return weights_history, train_loss_history[1:], test_loss_history[1:], change_loss_history[1:]
    
    
    
    
    #function to obtain the gradient 
    def GD_get_grad(self, weights):
        
        X = self.design_matrix(which = 'train')
        y = self.train_target
        b = weights
        gradient = np.dot(np.dot(X.T, X), b) - np.dot(X.T, y) 
            
        return gradient
    
    
    #function to obtain the gradient 
    def SGD_get_grad(self, X, y, weights):
        
        X = self.design_matrix(in_x = X, which = 'minibatch')
        y = y
        b = weights
        gradient = np.dot(np.dot(X.T, X), b) - np.dot(X.T, y) 
            
        return gradient
    
    
    # function to calculate the loss    
    def loss(self, y_pred, n, which, y_true=None):
        
        if self.optimizer == "MSE":
            c = 1/n
        else:
            c = 1/2
            
        if which == 'train':
            loss = c * np.sum(np.square(self.train_target - y_pred))

        if which == 'test':           
            loss = c * np.sum(np.square(self.test_target - y_pred))

        if which == 'all':   
            loss = c * np.sum(np.square(self.target - y_pred))
            
        if which == 'minibatch':
            loss = c * np.sum(np.square(y_true - y_pred))

        return loss
    
    
    #function to get the predicted values for a set of weights
    def predict(self, weights, X = None, which = ["train", "test", "all", 'minibatch']):
        
        if (which != 'all') and (which != 'minibatch'):
            X = self.design_matrix(which = which)
            Y_hat = np.dot(X, weights)
        if which == 'all':
            X = self.design_matrix(which = which)
            Y_hat = np.dot(X,  self.final_weights)
        if which == 'minibatch':
            X = self.design_matrix(in_x = X, which = which)
            Y_hat = np.dot(X,  weights)
            
        return Y_hat

            
        
        
    
    #function which returns the design matrix of X
    def design_matrix(self, in_x = None, which = ["train", "test", "all", 'minibatch']):
        
        
        if which == 'train':
            
            intercept_ind = np.array([1]*self.train_data.shape[0])
            desmat = np.concatenate((intercept_ind[:,None], self.train_data), axis = 1)
            
        if which == 'test':
            
            intercept_ind = np.array([1]*self.test_data.shape[0])
            desmat = np.concatenate((intercept_ind[:,None], self.test_data), axis = 1)
            
        if which == 'all':
        
            intercept_ind = np.array([1]*self.num_obs)
            desmat = np.concatenate((intercept_ind[:,None], self.data), axis = 1)
            
        if which == 'minibatch':
            intercept_ind = np.array([1]*in_x.shape[0])
            desmat = np.concatenate((intercept_ind[:,None], in_x), axis = 1)
            
        return desmat
    
    
    def plot_loss(self):

        plt.plot(self.its, self.train_loss_history[1:], lw=2, label = 'training loss')
        plt.plot(self.its, self.test_loss_history[1:], lw=2, label = 'testing loss')
        plt.ylabel(self.optimizer)
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()

        
    
    
    
    
    

        
        