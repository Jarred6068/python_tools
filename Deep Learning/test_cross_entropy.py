# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:31:30 2022

@author: Bruin
"""
import numpy as np
from keras import losses as l
y_true = np.array([[0, 1, 0], [0, 0, 1], [1,0,0]])
y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.75, 0.1, 0.1]])
# Using 'auto'/'sum_over_batch_size' reduction type.
cce = l.CategoricalCrossentropy(reduction = 'sum_over_batch_size')
print(cce(y_true, y_pred).numpy())

def cce2(y_pred, y):
    """
    1/n_samples*sum_samples(sum_output(-y_k*log(y_pred_k)))
    
    """
    m = y.shape[0]
    #epsilon = 1e-12
    #predictions = np.clip(y_pred, epsilon, 1. - epsilon)
    #ce = -np.sum(y_pred*np.log(predictions))/m
    ce = (1/m)*np.sum(np.sum(-y * np.log(y_pred+1e-16), axis = 1))
    return ce

print(cce2(y_pred, y_true))



    # def get_gradients(self, x, y):
    #     O_mat = self.layers[1].net

    #     grad_hidden_out = np.zeros([self.layers[1].input_dim, self.layers[1].units])
    #     delta_k_store = []
    #     for k in range(self.layers[1].units):
    #         dL_dnetk = np.zeros(O_mat.shape)
    #         for i in range(self.layers[1].units):
    #             dL_dOj = ((-1*y[:,i])/(1/O_mat[:,i]))
    #             if i == k:
    #                 dOj_dnetk = O_mat[:,k]*(1- O_mat[:,k])
    #             else:
    #                 dOj_dnetk = -O_mat[:,i]*(O_mat[:,k])
                    
    #             dL_dnetk[:, i] = dL_dOj * dOj_dnetk
                
    #         delta_k = dL_dnetk.sum(axis = 1)
    #         delta_k_store.append(delta_k)
    #         for j in range(self.layers[1].input_dim):
    #             grad_hidden_out[j,k] = np.inner(delta_k, nn.layers[1].inputs[:,j])
                
            
    #     # 2. calculate gradients for the input-to-hidden layers. 10 points
    #     h_mat = self.layers[0].net
    #     grad_input_hidden = np.zeros([self.layers[0].input_dim, self.layers[0].units])
    #     delta_i_store = []
    #     for j in range(self.layers[0].units):
    #         dL_dnetj = np.zeros(O_mat.shape)
    #         for k in range(self.layers[1].units):
    #             dL_dnetj[:,k] = self.layers[1].W[j,k]*delta_k_store[k]*dsigm(h_mat[:,j])
            
    #         delta_i = dL_dnetj.sum(axis = 1)
    #         delta_i_store.append(delta_i)
    #         for i in range(self.layers[0].input_dim):
    #             grad_input_hidden[i,j] = np.inner(delta_i, x[:,i])
                
        
    #     return grad_hidden_out, grad_input_hidden